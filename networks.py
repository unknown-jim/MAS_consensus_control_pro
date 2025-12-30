"""
神经网络模型 - CTDE 架构
Centralized Training Decentralized Execution

核心设计:
- Actor: 分散式，使用局部观测 + 邻居广播状态
- Critic: 集中式，使用全局状态 + 联合动作
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Optional, Tuple

from config import (
    STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_AGENTS, NUM_FOLLOWERS,
    LOG_STD_MIN, LOG_STD_MAX, U_SCALE, TH_SCALE,
    ACTOR_NUM_LAYERS, CRITIC_NUM_LAYERS,
    USE_NEIGHBOR_INFO, NEIGHBOR_AGGREGATION
)


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = HIDDEN_DIM, 
                 num_layers: int = 2, activation: type = nn.ReLU):
        super().__init__()
        
        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeighborAggregator(nn.Module):
    """邻居信息聚合器"""
    
    def __init__(self, obs_dim: int = STATE_DIM, hidden_dim: int = HIDDEN_DIM, 
                 aggregation: str = 'attention'):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.aggregation = aggregation
        
        self.neighbor_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        if aggregation == 'attention':
            self.query = nn.Linear(hidden_dim, hidden_dim)
            self.key = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, hidden_dim)
            self.scale = hidden_dim ** 0.5
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, self_feat: torch.Tensor, neighbor_obs: torch.Tensor, 
                neighbor_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            self_feat: 自身特征 (batch, hidden_dim)
            neighbor_obs: 邻居观测 (batch, max_neighbors, obs_dim)
            neighbor_mask: 邻居掩码 (batch, max_neighbors)
        
        Returns:
            aggregated: (batch, hidden_dim)
        """
        neighbor_feat = self.neighbor_encoder(neighbor_obs)
        
        if self.aggregation == 'attention':
            q = self.query(self_feat).unsqueeze(1)
            k = self.key(neighbor_feat)
            v = self.value(neighbor_feat)
            
            scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
            
            if neighbor_mask is not None:
                scores = scores.masked_fill(~neighbor_mask.unsqueeze(1), float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
            aggregated = torch.bmm(attn_weights, v).squeeze(1)
        
        else:  # mean aggregation
            if neighbor_mask is not None:
                neighbor_feat = neighbor_feat * neighbor_mask.unsqueeze(-1)
                num_neighbors = neighbor_mask.sum(dim=1, keepdim=True).clamp(min=1)
                aggregated = neighbor_feat.sum(dim=1) / num_neighbors
            else:
                aggregated = neighbor_feat.mean(dim=1)
        
        return self.output_proj(aggregated)


class DecentralizedActor(nn.Module):
    """
    分散式 Actor 网络
    
    执行时：每个智能体使用自己的局部观测 + 邻居的广播状态
    
    输出：
    - u: 控制量修正，范围 [-U_SCALE, U_SCALE]
    - threshold: 事件触发阈值，范围 [0, TH_SCALE]
    """
    
    def __init__(self, obs_dim: int = STATE_DIM, hidden_dim: int = HIDDEN_DIM, 
                 num_layers: int = ACTOR_NUM_LAYERS, 
                 use_neighbor_info: bool = USE_NEIGHBOR_INFO,
                 neighbor_aggregation: str = NEIGHBOR_AGGREGATION):
        super().__init__()
        
        self.use_neighbor_info = use_neighbor_info
        self.hidden_dim = hidden_dim
        
        self.self_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        if use_neighbor_info:
            self.neighbor_aggregator = NeighborAggregator(
                obs_dim, hidden_dim, neighbor_aggregation
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim)
            )
        
        # 控制量输出头
        self.u_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.u_log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 阈值输出头
        self.th_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.th_log_std = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.u_scale = U_SCALE
        self.th_scale = TH_SCALE
        self._eps = 1e-6
        self._log_u_scale = torch.log(torch.tensor(self.u_scale))
        self._log_th_scale = torch.log(torch.tensor(self.th_scale))
    
    def forward(self, local_obs: torch.Tensor, neighbor_obs: Optional[torch.Tensor] = None, 
                neighbor_mask: Optional[torch.Tensor] = None, 
                deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            local_obs: 局部观测 (batch, obs_dim)
            neighbor_obs: 邻居观测 (batch, max_neighbors, obs_dim)
            neighbor_mask: 邻居掩码 (batch, max_neighbors)
            deterministic: 是否确定性输出
        
        Returns:
            action: (batch, action_dim)
            log_prob: (batch, 1) 或 None
        """
        self_feat = self.self_encoder(local_obs)
        
        if self.use_neighbor_info and neighbor_obs is not None:
            neighbor_feat = self.neighbor_aggregator(self_feat, neighbor_obs, neighbor_mask)
            feat = self.fusion(torch.cat([self_feat, neighbor_feat], dim=-1))
        else:
            feat = self_feat
        
        u_mean = self.u_mean(feat)
        u_log_std = torch.clamp(self.u_log_std(feat), LOG_STD_MIN, LOG_STD_MAX)
        u_std = torch.exp(u_log_std)
        
        th_mean = self.th_mean(feat)
        th_log_std = torch.clamp(self.th_log_std(feat), LOG_STD_MIN, LOG_STD_MAX)
        th_std = torch.exp(th_log_std)
        
        if deterministic:
            u = torch.tanh(u_mean) * self.u_scale
            th = torch.sigmoid(th_mean) * self.th_scale
            log_prob = None
        else:
            u_dist = Normal(u_mean, u_std)
            th_dist = Normal(th_mean, th_std)
            
            u_sample = u_dist.rsample()
            th_sample = th_dist.rsample()
            
            u_tanh = torch.tanh(u_sample)
            u = u_tanh * self.u_scale
            
            th_sigmoid = torch.sigmoid(th_sample)
            th = th_sigmoid * self.th_scale
            
            log_prob_u = u_dist.log_prob(u_sample) - torch.log(
                torch.clamp(1.0 - u_tanh.pow(2), min=self._eps, max=1.0)
            ) - self._log_u_scale.to(u.device)
            
            log_prob_th = th_dist.log_prob(th_sample) - torch.log(
                torch.clamp(th_sigmoid * (1.0 - th_sigmoid), min=self._eps, max=0.25)
            ) - self._log_th_scale.to(th.device)
            
            log_prob = (log_prob_u + log_prob_th).sum(dim=-1, keepdim=True)
        
        action = torch.cat([u, th], dim=-1)
        return action, log_prob


class CentralizedCritic(nn.Module):
    """
    集中式 Critic 网络
    
    训练时：使用全局状态 + 联合动作进行评估
    
    输出：每个跟随者的 Q 值
    """
    
    def __init__(self, num_agents: int = NUM_AGENTS, obs_dim: int = STATE_DIM, 
                 action_dim: int = ACTION_DIM, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = CRITIC_NUM_LAYERS):
        super().__init__()
        
        self.num_agents = num_agents
        self.num_followers = num_agents - 1
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        global_obs_dim = num_agents * obs_dim
        joint_action_dim = self.num_followers * action_dim
        
        self.state_encoder = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.action_encoder = nn.Sequential(
            nn.Linear(joint_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.q_head = MLP(hidden_dim, self.num_followers, hidden_dim, num_layers=num_layers)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: 全局状态 (batch, num_agents, obs_dim)
            actions: 联合动作 (batch, num_followers, action_dim)
        
        Returns:
            q_values: (batch, num_followers)
        """
        batch_size = states.shape[0]
        
        global_obs = states.view(batch_size, -1)
        joint_action = actions.view(batch_size, -1)
        
        state_feat = self.state_encoder(global_obs)
        action_feat = self.action_encoder(joint_action)
        
        fused = self.fusion(torch.cat([state_feat, action_feat], dim=-1))
        
        return self.q_head(fused)
