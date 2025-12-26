"""
神经网络模型 - GAT 编码器、Actor、Critic（修复版）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

try:
    from torch_geometric.nn import GATConv
except ImportError:
    raise ImportError(
        "torch_geometric 未安装，请运行:\n"
        "pip install torch-geometric\n"
        "或参考: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    )

from config import (
    STATE_DIM, HIDDEN_DIM, ACTION_DIM,
    LOG_STD_MIN, LOG_STD_MAX, U_SCALE, TH_SCALE  # 🔧 添加导入
)


class TopologyAwareGATEncoder(nn.Module):
    """拓扑感知的图注意力编码器"""
    
    def __init__(self, in_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, num_heads=4):
        super(TopologyAwareGATEncoder, self).__init__()
        
        self.gat1 = GATConv(in_dim, hidden_dim, heads=num_heads, concat=True, 
                           add_self_loops=True, dropout=0.1)
        self.gat2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=num_heads, 
                           concat=True, add_self_loops=True, dropout=0.1)
        self.gat3 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, 
                           concat=False, add_self_loops=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim * num_heads)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.role_embedding = nn.Embedding(2, hidden_dim)
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, role_ids):
        h = F.elu(self.gat1(x, edge_index))
        h = self.norm1(h)
        
        h = F.elu(self.gat2(h, edge_index))
        h = self.norm2(h)
        
        h = self.gat3(h, edge_index)
        h = self.norm3(h)
        
        role_emb = self.role_embedding(role_ids)
        h = self.output_proj(torch.cat([h, role_emb], dim=-1))
        
        return h


class GaussianActor(nn.Module):
    """高斯策略 Actor 网络（修复版）"""
    
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, num_heads=4):
        super(GaussianActor, self).__init__()
        
        self.encoder = TopologyAwareGATEncoder(state_dim, hidden_dim, num_heads)
        
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
        
        # 🔧 使用配置中的缩放因子
        self.u_scale = U_SCALE
        self.th_scale = TH_SCALE
        
        # 数值稳定性常数
        self._eps = 1e-6
        
        # 🔧 预计算 log(scale) 以提高效率
        self._log_u_scale = torch.log(torch.tensor(self.u_scale))
        self._log_th_scale = torch.log(torch.tensor(self.th_scale))
    
    def forward(self, x, edge_index, role_ids, deterministic=False):
        feat = self.encoder(x, edge_index, role_ids)
        
        follower_mask = role_ids == 1
        follower_feat = feat[follower_mask]
        
        u_mean = self.u_mean(follower_feat)
        u_log_std = torch.clamp(self.u_log_std(follower_feat), LOG_STD_MIN, LOG_STD_MAX)
        u_std = torch.exp(u_log_std)
        
        th_mean = self.th_mean(follower_feat)
        th_log_std = torch.clamp(self.th_log_std(follower_feat), LOG_STD_MIN, LOG_STD_MAX)
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
            
            # 应用变换
            u_tanh = torch.tanh(u_sample)
            u = u_tanh * self.u_scale
            
            th_sigmoid = torch.sigmoid(th_sample)
            th = th_sigmoid * self.th_scale
            
            # 🔧 修复：正确计算 log_prob，考虑 scale 因子
            # 变换: u = u_scale * tanh(u_sample)
            # Jacobian: du/d(u_sample) = u_scale * (1 - tanh^2(u_sample))
            # log|Jacobian| = log(u_scale) + log(1 - tanh^2(u_sample))
            log_prob_u = u_dist.log_prob(u_sample) - torch.log(
                torch.clamp(1.0 - u_tanh.pow(2), min=self._eps, max=1.0)
            ) - self._log_u_scale.to(u.device)
            
            # 变换: th = th_scale * sigmoid(th_sample)
            # Jacobian: dth/d(th_sample) = th_scale * sigmoid * (1 - sigmoid)
            # log|Jacobian| = log(th_scale) + log(sigmoid) + log(1 - sigmoid)
            log_prob_th = th_dist.log_prob(th_sample) - torch.log(
                torch.clamp(th_sigmoid * (1.0 - th_sigmoid), min=self._eps, max=0.25)
            ) - self._log_th_scale.to(th.device)
            
            log_prob = (log_prob_u + log_prob_th).sum(dim=-1, keepdim=True)
        
        action = torch.cat([u, th], dim=-1)
        return action, log_prob, follower_mask


class SoftQNetwork(nn.Module):
    """Soft Q 网络"""
    
    def __init__(self, state_dim=STATE_DIM, hidden_dim=HIDDEN_DIM, 
                 action_dim=ACTION_DIM, num_heads=4):
        super(SoftQNetwork, self).__init__()
        
        self.encoder = TopologyAwareGATEncoder(state_dim, hidden_dim, num_heads)
        
        self.q_net = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x, edge_index, role_ids, action):
        feat = self.encoder(x, edge_index, role_ids)
        
        follower_mask = role_ids == 1
        follower_feat = feat[follower_mask]
        
        q_input = torch.cat([follower_feat, action], dim=-1)
        q_value = self.q_net(q_input)
        
        return q_value