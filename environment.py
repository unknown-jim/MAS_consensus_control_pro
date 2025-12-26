"""
领导者-跟随者多智能体系统环境 - 课程学习优化版（修复版）
"""
import torch
import math

from config import (
    DEVICE, STATE_DIM, DT,
    COMM_PENALTY_INIT, 
    THRESHOLD_MIN_INIT, THRESHOLD_MAX_INIT,
    THRESHOLD_MIN_FINAL, THRESHOLD_MAX_FINAL,
    COMM_BONUS_INIT,
    LEADER_AMPLITUDE, LEADER_OMEGA, LEADER_PHASE,
    POS_LIMIT, VEL_LIMIT,
    REWARD_MIN, REWARD_MAX, USE_SOFT_REWARD_SCALING,
    TH_SCALE  # 🔧 添加导入
)


class BatchedLeaderFollowerEnv:
    """完全向量化的批量环境 - 课程学习优化版（修复版）"""
    
    def __init__(self, topology, num_envs=64):
        self.topology = topology
        self.num_envs = num_envs
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.leader_id = topology.leader_id
        
        self.leader_amplitude = LEADER_AMPLITUDE
        self.leader_omega = LEADER_OMEGA
        self.leader_phase = LEADER_PHASE
        
        self.pos_limit = POS_LIMIT
        self.vel_limit = VEL_LIMIT
        self.reward_min = REWARD_MIN
        self.reward_max = REWARD_MAX
        self.use_soft_scaling = USE_SOFT_REWARD_SCALING
        
        # 控制器增益
        self.base_pos_gain = 5.0
        self.base_vel_gain = 2.5
        
        # 🔧 课程学习参数（由外部设置）
        self.comm_penalty = COMM_PENALTY_INIT
        self.threshold_min = THRESHOLD_MIN_INIT
        self.threshold_max = THRESHOLD_MAX_INIT
        self.comm_bonus = COMM_BONUS_INIT
        
        # 🔧 课程学习进度（0-1）
        self.curriculum_progress = 0.0
        
        # 🔧 阈值缩放因子（与 networks.py 保持一致）
        self.th_scale = TH_SCALE
        
        self.role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self.role_ids[1:] = 1
        
        self._precompute_neighbor_info()
        
        # 预分配状态张量
        self.positions = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.velocities = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_pos = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_vel = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.t = torch.zeros(num_envs, device=DEVICE)
        
        self._prev_error = None
        self.reset()
    
    def set_curriculum_params(self, comm_penalty, threshold_min, threshold_max, comm_bonus, progress):
        """
        设置课程学习参数
        
        Args:
            comm_penalty: 通信惩罚系数
            threshold_min: 阈值下界
            threshold_max: 阈值上界
            comm_bonus: 通信奖励系数
            progress: 课程进度 [0, 1]
        """
        self.comm_penalty = comm_penalty
        self.threshold_min = threshold_min
        self.threshold_max = threshold_max
        self.comm_bonus = comm_bonus
        self.curriculum_progress = progress
    
    def set_comm_penalty(self, penalty):
        """设置当前通信惩罚系数（兼容旧接口）"""
        self.comm_penalty = penalty
    
    def _precompute_neighbor_info(self):
        """预计算邻居聚合矩阵"""
        self.adj_matrix = torch.zeros(self.num_agents, self.num_agents, device=DEVICE)
        edge_index = self.topology.edge_index
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            self.adj_matrix[dst, src] = 1.0
        
        in_degree = self.adj_matrix.sum(dim=1)
        self.degree_matrix = torch.diag(in_degree)
        self.laplacian = self.degree_matrix - self.adj_matrix
        
        in_degree_safe = in_degree.clamp(min=1.0)
        self.norm_adj_matrix = self.adj_matrix / in_degree_safe.unsqueeze(1)
        
        self.pinning_gains = torch.zeros(self.num_agents, device=DEVICE)
        for f in self.topology.pinned_followers:
            self.pinning_gains[f] = 2.0
    
    def _leader_state_batch(self, t):
        """批量计算领导者状态"""
        pos = self.leader_amplitude * torch.sin(self.leader_omega * t + self.leader_phase)
        vel = self.leader_amplitude * self.leader_omega * torch.cos(self.leader_omega * t + self.leader_phase)
        return pos, vel
    
    def reset(self, env_ids=None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=DEVICE)
        
        num_reset = len(env_ids) if isinstance(env_ids, torch.Tensor) else self.num_envs
        
        self.t[env_ids] = 0.0
        
        leader_pos, leader_vel = self._leader_state_batch(self.t[env_ids])
        
        self.positions[env_ids, 0] = leader_pos
        self.velocities[env_ids, 0] = leader_vel
        
        # 🔧 初始位置根据课程进度调整
        init_pos_std = 0.2 + 0.3 * self.curriculum_progress
        init_vel_std = 0.05 + 0.1 * self.curriculum_progress
        
        self.positions[env_ids, 1:] = leader_pos.unsqueeze(1) + torch.randn(
            num_reset, self.num_followers, device=DEVICE
        ) * init_pos_std
        self.velocities[env_ids, 1:] = leader_vel.unsqueeze(1) + torch.randn(
            num_reset, self.num_followers, device=DEVICE
        ) * init_vel_std
        
        self.last_broadcast_pos[env_ids] = self.positions[env_ids].clone()
        self.last_broadcast_vel[env_ids] = self.velocities[env_ids].clone()
        
        self._prev_error = None
        
        return self._get_state()
    
    def _get_state(self):
        """构建观测状态"""
        state = torch.zeros(self.num_envs, self.num_agents, STATE_DIM, device=DEVICE)
        
        neighbor_avg_pos = torch.matmul(self.last_broadcast_pos, self.norm_adj_matrix.T)
        neighbor_avg_vel = torch.matmul(self.last_broadcast_vel, self.norm_adj_matrix.T)
        
        pos_error = self.positions - neighbor_avg_pos
        vel_error = self.velocities - neighbor_avg_vel
        
        state[:, :, 0] = pos_error / (self.pos_limit + 1e-6)
        state[:, :, 1] = vel_error / (self.vel_limit + 1e-6)
        state[:, :, 2] = self.positions / (self.pos_limit + 1e-6)
        state[:, :, 3] = self.velocities / (self.vel_limit + 1e-6)
        
        return state
    
    def _compute_base_control(self):
        """计算基础一致性控制"""
        follower_pos = self.last_broadcast_pos[:, 1:]
        follower_vel = self.last_broadcast_vel[:, 1:]
        
        leader_pos = self.last_broadcast_pos[:, 0:1]
        leader_vel = self.last_broadcast_vel[:, 0:1]
        
        follower_adj = self.adj_matrix[1:, 1:]
        follower_degree = follower_adj.sum(dim=1, keepdim=True).clamp(min=1.0)
        
        neighbor_pos_sum = torch.matmul(follower_pos, follower_adj.T)
        neighbor_vel_sum = torch.matmul(follower_vel, follower_adj.T)
        
        pos_consensus_error = follower_pos * follower_degree.T - neighbor_pos_sum
        vel_consensus_error = follower_vel * follower_degree.T - neighbor_vel_sum
        
        pinning_gains_followers = self.pinning_gains[1:]
        pos_pinning_error = (follower_pos - leader_pos) * pinning_gains_followers
        vel_pinning_error = (follower_vel - leader_vel) * pinning_gains_followers
        
        base_control = (
            -self.base_pos_gain * (pos_consensus_error + pos_pinning_error)
            -self.base_vel_gain * (vel_consensus_error + vel_pinning_error)
        )
        
        return base_control
    
    def _scale_reward_batch(self, reward):
        """批量奖励缩放"""
        if self.use_soft_scaling:
            mid = (self.reward_max + self.reward_min) / 2
            scale = (self.reward_max - self.reward_min) / 2
            normalized = (reward - mid) / (scale + 1e-8)
            return mid + scale * torch.tanh(normalized)
        else:
            return torch.clamp(reward, self.reward_min, self.reward_max)
    
    def step(self, action):
        """批量执行一步"""
        self.t += DT
        
        # 更新领导者
        leader_pos, leader_vel = self._leader_state_batch(self.t)
        self.positions[:, 0] = leader_pos
        self.velocities[:, 0] = leader_vel
        
        # 解析动作
        delta_u = action[:, :, 0] * 2.0
        raw_threshold = action[:, :, 1]
        
        # 🔧 修复：线性映射，不再使用 sigmoid
        # raw_threshold 范围是 [0, TH_SCALE]，直接归一化到 [0, 1]
        normalized_threshold = raw_threshold / self.th_scale
        # 确保归一化值在 [0, 1] 范围内
        normalized_threshold = normalized_threshold.clamp(0.0, 1.0)
        # 映射到当前课程阶段的阈值范围
        threshold = self.threshold_min + (self.threshold_max - self.threshold_min) * normalized_threshold
        threshold = threshold.clamp(min=0.001, max=0.5)
        
        # 计算总控制
        base_u = self._compute_base_control()
        total_u = base_u + delta_u
        total_u = torch.clamp(total_u, -20.0, 20.0)
        
        # 跟随者动力学
        follower_pos = self.positions[:, 1:]
        follower_vel = self.velocities[:, 1:]
        
        nonlinear_term = 0.2 * torch.sin(follower_pos) - 0.1 * follower_vel
        acc = total_u + nonlinear_term
        
        new_vel = torch.clamp(follower_vel + acc * DT, -self.vel_limit, self.vel_limit)
        new_pos = torch.clamp(follower_pos + new_vel * DT, -self.pos_limit, self.pos_limit)
        
        self.positions[:, 1:] = new_pos
        self.velocities[:, 1:] = new_vel
        
        # 事件触发通信
        trigger_error = torch.abs(new_pos - self.last_broadcast_pos[:, 1:])
        is_triggered = trigger_error > threshold
        
        self.last_broadcast_pos[:, 1:] = torch.where(
            is_triggered, self.positions[:, 1:], self.last_broadcast_pos[:, 1:]
        )
        self.last_broadcast_vel[:, 1:] = torch.where(
            is_triggered, self.velocities[:, 1:], self.last_broadcast_vel[:, 1:]
        )
        self.last_broadcast_pos[:, 0] = self.positions[:, 0]
        self.last_broadcast_vel[:, 0] = self.velocities[:, 0]
        
        # ==================== 计算奖励（优化版）====================
        pos_error = torch.abs(self.positions[:, 1:] - self.positions[:, 0:1])
        vel_error = torch.abs(self.velocities[:, 1:] - self.velocities[:, 0:1])
        
        tracking_error = pos_error.mean(dim=1) + 0.5 * vel_error.mean(dim=1)
        
        # 跟踪奖励（主要奖励）
        tracking_reward = torch.exp(-tracking_error) * 2.5 - 1.0
        
        # 改进奖励
        improvement_bonus = torch.zeros_like(tracking_error)
        if self._prev_error is not None:
            improvement = self._prev_error - tracking_error
            improvement_bonus = torch.clamp(improvement * 3.0, -0.5, 0.5)
        self._prev_error = tracking_error.detach().clone()
        
        # 通信率
        comm_rate = is_triggered.float().mean(dim=1)
        
        # 通信惩罚
        comm_penalty = comm_rate * self.comm_penalty
        
        # 通信奖励（早期阶段）
        comm_reward = comm_rate * self.comm_bonus
        
        # 一致性奖励
        max_error = pos_error.max(dim=1)[0]
        consensus_bonus = torch.where(
            max_error < 0.3,
            torch.ones_like(max_error) * 0.2,
            torch.zeros_like(max_error)
        )
        
        # 总奖励
        raw_reward = (
            tracking_reward +
            improvement_bonus +
            comm_reward -
            comm_penalty +
            consensus_bonus
        )
        rewards = self._scale_reward_batch(raw_reward)
        
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        
        infos = {
            'tracking_error': tracking_error,
            'comm_rate': comm_rate,
            'leader_pos': self.positions[:, 0],
            'leader_vel': self.velocities[:, 0],
            'avg_follower_pos': self.positions[:, 1:].mean(dim=1),
            'threshold_mean': threshold.mean(),
            'threshold_min': self.threshold_min,
            'threshold_max': self.threshold_max,
            'comm_penalty_coef': self.comm_penalty,
            'comm_bonus_coef': self.comm_bonus,
            'curriculum_progress': self.curriculum_progress,
            'tracking_reward': tracking_reward.mean(),
            'comm_reward': comm_reward.mean(),
            'comm_penalty_value': comm_penalty.mean(),
        }
        
        return self._get_state(), rewards, dones, infos


class LeaderFollowerMASEnv:
    """单环境版本"""
    
    def __init__(self, topology):
        self.batched_env = BatchedLeaderFollowerEnv(topology, num_envs=1)
        self.topology = topology
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.role_ids = self.batched_env.role_ids
    
    @property
    def positions(self):
        return self.batched_env.positions[0]
    
    @property
    def velocities(self):
        return self.batched_env.velocities[0]
    
    @property
    def t(self):
        return self.batched_env.t[0].item()
    
    def set_curriculum_params(self, comm_penalty, threshold_min, threshold_max, comm_bonus, progress):
        """设置课程学习参数"""
        self.batched_env.set_curriculum_params(
            comm_penalty, threshold_min, threshold_max, comm_bonus, progress
        )
    
    def set_comm_penalty(self, penalty):
        """兼容旧接口"""
        self.batched_env.set_comm_penalty(penalty)
    
    def reset(self):
        state = self.batched_env.reset()
        return state[0]
    
    def step(self, action):
        action_batched = action.unsqueeze(0)
        states, rewards, dones, infos = self.batched_env.step(action_batched)
        info = {k: (v[0].item() if isinstance(v, torch.Tensor) and v.dim() > 0 else 
                    v.item() if isinstance(v, torch.Tensor) else v) 
                for k, v in infos.items()}
        return states[0], rewards[0].item(), dones[0].item(), info