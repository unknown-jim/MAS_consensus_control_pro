"""
SAC 智能体 - CTDE 架构
"""
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from config import (
    DEVICE, STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_AGENTS,
    LEARNING_RATE, ALPHA_LR, GAMMA, TAU, BATCH_SIZE,
    INIT_ALPHA, GRADIENT_STEPS, AUTO_ALPHA,
    USE_NEIGHBOR_INFO, MAX_NEIGHBORS, WARMUP_STEPS
)
from buffer import OptimizedReplayBuffer
from networks import DecentralizedActor, CentralizedCritic


class SACAgent:
    """CTDE-SAC 智能体"""
    
    def __init__(self, topology, auto_entropy: bool = AUTO_ALPHA, use_amp: bool = True):
        self.topology = topology
        self.num_followers = topology.num_followers
        self.num_agents = topology.num_agents
        self.auto_entropy = auto_entropy
        self.use_amp = use_amp and torch.cuda.is_available()
        self.use_neighbor_info = USE_NEIGHBOR_INFO
        self.warmup_steps = WARMUP_STEPS
        
        # 预计算邻居信息
        if self.use_neighbor_info:
            self._precompute_neighbor_info()
        
        # 网络初始化
        self.actor = DecentralizedActor(
            STATE_DIM, HIDDEN_DIM,
            use_neighbor_info=USE_NEIGHBOR_INFO
        ).to(DEVICE)
        
        self.q1 = CentralizedCritic(NUM_AGENTS, STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
        self.q2 = CentralizedCritic(NUM_AGENTS, STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
        self.q1_target = CentralizedCritic(NUM_AGENTS, STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
        self.q2_target = CentralizedCritic(NUM_AGENTS, STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(DEVICE)
        
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        for param in self.q1_target.parameters():
            param.requires_grad = False
        for param in self.q2_target.parameters():
            param.requires_grad = False
        
        # AMP 混合精度
        if self.use_amp:
            self.scaler = torch.amp.GradScaler('cuda')
            print(f"CTDE-SAC with AMP enabled")
        else:
            self.scaler = None
            print(f"CTDE-SAC initialized")
        
        # 熵系数
        self.target_entropy = -float(ACTION_DIM) * 0.5
        self.log_alpha = torch.tensor(np.log(INIT_ALPHA), requires_grad=True, device=DEVICE)
        self.alpha = self.log_alpha.exp().item()
        self.alpha_min = 0.01
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LEARNING_RATE)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LEARNING_RATE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=ALPHA_LR)
        
        # 经验回放
        self.buffer = OptimizedReplayBuffer(num_agents=NUM_AGENTS)
        
        # 统计
        self.last_losses = {'q1': 0, 'q2': 0, 'actor': 0, 'alpha': INIT_ALPHA}
        self.update_count = 0
        self.total_steps = 0
    
    def _precompute_neighbor_info(self) -> None:
        """预计算每个跟随者的邻居索引"""
        self.neighbor_indices = {}
        self.max_neighbors = MAX_NEIGHBORS
        
        for follower_id in range(1, self.num_agents):
            neighbors = self.topology.get_neighbors(follower_id)
            self.neighbor_indices[follower_id] = neighbors[:self.max_neighbors]
    
    def _get_neighbor_obs(self, states: torch.Tensor, batch_size: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """获取邻居观测"""
        if not self.use_neighbor_info:
            return None, None
        
        neighbor_obs = torch.zeros(
            batch_size, self.num_followers, self.max_neighbors, STATE_DIM,
            device=DEVICE
        )
        neighbor_mask = torch.zeros(
            batch_size, self.num_followers, self.max_neighbors,
            dtype=torch.bool, device=DEVICE
        )
        
        for i, follower_id in enumerate(range(1, self.num_agents)):
            neighbors = self.neighbor_indices.get(follower_id, [])
            for j, neighbor_id in enumerate(neighbors):
                if j >= self.max_neighbors:
                    break
                neighbor_obs[:, i, j, :] = states[:, neighbor_id, :]
                neighbor_mask[:, i, j] = True
        
        neighbor_obs = neighbor_obs.view(-1, self.max_neighbors, STATE_DIM)
        neighbor_mask = neighbor_mask.view(-1, self.max_neighbors)
        
        return neighbor_obs, neighbor_mask
    
    def _get_follower_obs(self, states: torch.Tensor, batch_size: int) -> torch.Tensor:
        """获取跟随者观测"""
        follower_obs = states[:, 1:, :]
        return follower_obs.reshape(-1, STATE_DIM)
    
    def _forward_actor(self, follower_obs: torch.Tensor, neighbor_obs: Optional[torch.Tensor],
                       neighbor_mask: Optional[torch.Tensor], deterministic: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Actor 前向传播（统一 AMP 处理）"""
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                return self.actor(follower_obs, neighbor_obs=neighbor_obs, 
                                  neighbor_mask=neighbor_mask, deterministic=deterministic)
        return self.actor(follower_obs, neighbor_obs=neighbor_obs,
                          neighbor_mask=neighbor_mask, deterministic=deterministic)
    
    @torch.no_grad()
    def select_action(self, state: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """选择动作"""
        is_batched = state.dim() == 3
        
        if not is_batched:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        follower_obs = self._get_follower_obs(state, batch_size)
        neighbor_obs, neighbor_mask = self._get_neighbor_obs(state, batch_size)
        
        action, _ = self._forward_actor(follower_obs, neighbor_obs, neighbor_mask, deterministic)
        action = action.view(batch_size, self.num_followers, ACTION_DIM)
        
        if not is_batched:
            action = action.squeeze(0)
        
        return action.float()
    
    def store_transitions_batch(self, states: torch.Tensor, actions: torch.Tensor, 
                                rewards: torch.Tensor, next_states: torch.Tensor, 
                                dones: torch.Tensor) -> None:
        """批量存储经验"""
        self.buffer.push_batch(states, actions, rewards, next_states, dones)
        self.total_steps += states.shape[0]
    
    def _update_critic(self, states: torch.Tensor, actions: torch.Tensor, 
                       target_q: torch.Tensor, optimizer: optim.Optimizer, 
                       critic: CentralizedCritic) -> float:
        """更新单个 Critic"""
        optimizer.zero_grad(set_to_none=True)
        
        if self.use_amp:
            with torch.amp.autocast('cuda'):
                q_curr = critic(states, actions).mean(dim=1, keepdim=True)
                loss = F.mse_loss(q_curr.float(), target_q)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            self.scaler.step(optimizer)
        else:
            q_curr = critic(states, actions).mean(dim=1, keepdim=True)
            loss = F.mse_loss(q_curr, target_q)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            optimizer.step()
        
        return loss.item()
    
    def update(self, batch_size: int = BATCH_SIZE, gradient_steps: int = GRADIENT_STEPS) -> Dict[str, float]:
        """更新网络"""
        if self.total_steps < self.warmup_steps:
            return {}
        
        if not self.buffer.is_ready(batch_size):
            return {}
        
        total_q1_loss = 0
        total_q2_loss = 0
        total_actor_loss = 0
        
        for _ in range(gradient_steps):
            self.update_count += 1
            
            states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
            
            follower_obs = self._get_follower_obs(states, batch_size)
            follower_next_obs = self._get_follower_obs(next_states, batch_size)
            
            neighbor_obs, neighbor_mask = self._get_neighbor_obs(states, batch_size)
            neighbor_next_obs, neighbor_next_mask = self._get_neighbor_obs(next_states, batch_size)
            
            # 计算目标 Q 值
            with torch.no_grad():
                next_actions, next_log_probs = self._forward_actor(
                    follower_next_obs, neighbor_next_obs, neighbor_next_mask
                )
                next_actions_reshaped = next_actions.view(batch_size, self.num_followers, ACTION_DIM)
                
                q1_next = self.q1_target(next_states, next_actions_reshaped)
                q2_next = self.q2_target(next_states, next_actions_reshaped)
                
                q_next = torch.min(q1_next, q2_next).mean(dim=1, keepdim=True)
                next_log_probs = next_log_probs.view(batch_size, self.num_followers).mean(dim=1, keepdim=True)
                
                target_q = rewards.unsqueeze(1) + GAMMA * (1 - dones.unsqueeze(1)) * (
                    q_next - self.alpha * next_log_probs
                )
                target_q = target_q.float()
            
            # Critic 更新
            total_q1_loss += self._update_critic(states, actions, target_q, self.q1_optimizer, self.q1)
            total_q2_loss += self._update_critic(states, actions, target_q, self.q2_optimizer, self.q2)
            
            # Actor 更新
            self.actor_optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    new_actions, log_probs = self.actor(
                        follower_obs, neighbor_obs=neighbor_obs, neighbor_mask=neighbor_mask
                    )
                    new_actions_reshaped = new_actions.view(batch_size, self.num_followers, ACTION_DIM)
                    
                    q1_new = self.q1(states, new_actions_reshaped)
                    q2_new = self.q2(states, new_actions_reshaped)
                    q_new = torch.min(q1_new, q2_new)
                    
                    actor_loss = (self.alpha * log_probs - q_new.view(-1, 1)).mean()
                
                self.scaler.scale(actor_loss).backward()
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.scaler.step(self.actor_optimizer)
            else:
                new_actions, log_probs = self.actor(
                    follower_obs, neighbor_obs=neighbor_obs, neighbor_mask=neighbor_mask
                )
                new_actions_reshaped = new_actions.view(batch_size, self.num_followers, ACTION_DIM)
                
                q1_new = self.q1(states, new_actions_reshaped)
                q2_new = self.q2(states, new_actions_reshaped)
                q_new = torch.min(q1_new, q2_new)
                
                actor_loss = (self.alpha * log_probs - q_new.view(-1, 1)).mean()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                self.actor_optimizer.step()
            
            # Alpha 更新
            if self.auto_entropy:
                self.alpha_optimizer.zero_grad(set_to_none=True)
                alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = max(self.log_alpha.exp().item(), self.alpha_min)
            
            # 目标网络软更新
            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)
            
            if self.use_amp:
                self.scaler.update()
            
            total_actor_loss += actor_loss.item()
        
        self.last_losses = {
            'q1': total_q1_loss / gradient_steps,
            'q2': total_q2_loss / gradient_steps,
            'actor': total_actor_loss / gradient_steps,
            'alpha': self.alpha
        }
        
        return self.last_losses
    
    @torch.no_grad()
    def _soft_update(self, source: torch.nn.Module, target: torch.nn.Module) -> None:
        """软更新目标网络"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.lerp_(param.data, TAU)
    
    def save(self, path: str) -> None:
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'log_alpha': self.log_alpha,
            'update_count': self.update_count,
            'total_steps': self.total_steps,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str) -> None:
        """加载模型"""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.q1.load_state_dict(checkpoint['q1'])
        self.q2.load_state_dict(checkpoint['q2'])
        self.q1_target.load_state_dict(checkpoint['q1_target'])
        self.q2_target.load_state_dict(checkpoint['q2_target'])
        if 'log_alpha' in checkpoint:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha = self.log_alpha.exp().item()
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
        print(f"Model loaded from {path}")
