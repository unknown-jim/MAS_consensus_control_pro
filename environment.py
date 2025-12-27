"""
领导者-跟随者多智能体系统环境 - 自适应奖励设计

奖励设计核心思想：
1. 跟踪惩罚（饱和型）: -tanh(error * scale) * max_penalty
   - 小误差时近似线性，大误差时饱和，避免梯度爆炸
   
2. 通信惩罚（自适应权重）: -comm_rate * base * exp(-error * decay)
   - 误差大时：通信权重低，专注于减小误差
   - 误差小时：通信权重高，开始优化通信效率

3. 改进奖励：鼓励误差持续减小
"""
import torch
import numpy as np

from config import (
    DEVICE, STATE_DIM, DT,
    THRESHOLD_MIN, THRESHOLD_MAX,
    LEADER_AMPLITUDE, LEADER_OMEGA, LEADER_PHASE,
    POS_LIMIT, VEL_LIMIT,
    REWARD_MIN, REWARD_MAX, USE_SOFT_REWARD_SCALING,
    TH_SCALE,
    # 奖励参数
    TRACKING_ERROR_SCALE, TRACKING_PENALTY_MAX,
    COMM_PENALTY_BASE, COMM_WEIGHT_DECAY,
    IMPROVEMENT_SCALE, IMPROVEMENT_CLIP,
    # 随机化配置
    ENABLE_RANDOMIZATION,
    LEADER_AMP_RANGE, LEADER_OMEGA_RANGE, LEADER_PHASE_RANGE,
    LEADER_TRAJECTORY_TYPES,
    FOLLOWER_POS_INIT_STD_RANGE, FOLLOWER_VEL_INIT_STD_RANGE
)


class BatchedLeaderFollowerEnv:
    """
    完全向量化的批量环境 - 自适应奖励设计
    
    特性：
    1. 领导者动力学随机化
    2. 跟随者初始状态随机化
    3. 自适应奖励：误差大时专注跟踪，误差小时优化通信
    """
    
    def __init__(self, topology, num_envs=64, enable_randomization=ENABLE_RANDOMIZATION):
        self.topology = topology
        self.num_envs = num_envs
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.leader_id = topology.leader_id
        self.enable_randomization = enable_randomization
        
        # 随机化范围
        self.amp_range = LEADER_AMP_RANGE
        self.omega_range = LEADER_OMEGA_RANGE
        self.phase_range = LEADER_PHASE_RANGE
        self.trajectory_types = LEADER_TRAJECTORY_TYPES
        self.pos_std_range = FOLLOWER_POS_INIT_STD_RANGE
        self.vel_std_range = FOLLOWER_VEL_INIT_STD_RANGE
        
        # 领导者参数
        self.leader_amplitude = torch.full((num_envs,), LEADER_AMPLITUDE, device=DEVICE)
        self.leader_omega = torch.full((num_envs,), LEADER_OMEGA, device=DEVICE)
        self.leader_phase = torch.full((num_envs,), LEADER_PHASE, device=DEVICE)
        
        # 轨迹类型
        self.trajectory_type_ids = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
        self.type_to_id = {'sine': 0, 'cosine': 1, 'mixed': 2, 'chirp': 3}
        self.id_to_type = {v: k for k, v in self.type_to_id.items()}
        
        # 环境参数
        self.pos_limit = POS_LIMIT
        self.vel_limit = VEL_LIMIT
        self.reward_min = REWARD_MIN
        self.reward_max = REWARD_MAX
        self.use_soft_scaling = USE_SOFT_REWARD_SCALING
        
        # 奖励参数
        self.tracking_error_scale = TRACKING_ERROR_SCALE
        self.tracking_penalty_max = TRACKING_PENALTY_MAX
        self.comm_penalty_base = COMM_PENALTY_BASE
        self.comm_weight_decay = COMM_WEIGHT_DECAY
        self.improvement_scale = IMPROVEMENT_SCALE
        self.improvement_clip = IMPROVEMENT_CLIP
        
        # 控制器增益
        self.base_pos_gain = 5.0
        self.base_vel_gain = 2.5
        
        # 通信参数
        self.threshold_min = THRESHOLD_MIN
        self.threshold_max = THRESHOLD_MAX
        self.th_scale = TH_SCALE
        
        # 角色标识
        self.role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self.role_ids[1:] = 1
        
        # 预计算邻居信息
        self._precompute_neighbor_info()
        
        # 预分配状态张量
        self.positions = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.velocities = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_pos = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_vel = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.t = torch.zeros(num_envs, device=DEVICE)
        
        self._prev_error = None
        self.reset()
    
    def _precompute_neighbor_info(self):
        """预计算邻居聚合矩阵"""
        self.adj_matrix = torch.zeros(self.num_agents, self.num_agents, device=DEVICE)
        edge_index = self.topology.edge_index
        
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            self.adj_matrix[dst, src] = 1.0
        
        in_degree = self.adj_matrix.sum(dim=1)
        in_degree_safe = in_degree.clamp(min=1.0)
        self.norm_adj_matrix = self.adj_matrix / in_degree_safe.unsqueeze(1)
        
        self.pinning_gains = torch.zeros(self.num_agents, device=DEVICE)
        for f in self.topology.pinned_followers:
            self.pinning_gains[f] = 2.0
    
    def _randomize_leader_dynamics(self, env_ids):
        """随机化领导者动力学参数"""
        if isinstance(env_ids, torch.Tensor):
            num_envs = len(env_ids)
        else:
            num_envs = self.num_envs
            env_ids = torch.arange(self.num_envs, device=DEVICE)
        
        self.leader_amplitude[env_ids] = torch.rand(num_envs, device=DEVICE) * \
            (self.amp_range[1] - self.amp_range[0]) + self.amp_range[0]
        
        self.leader_omega[env_ids] = torch.rand(num_envs, device=DEVICE) * \
            (self.omega_range[1] - self.omega_range[0]) + self.omega_range[0]
        
        self.leader_phase[env_ids] = torch.rand(num_envs, device=DEVICE) * \
            (self.phase_range[1] - self.phase_range[0]) + self.phase_range[0]
        
        random_types = np.random.choice(
            [self.type_to_id[t] for t in self.trajectory_types], 
            size=num_envs
        )
        self.trajectory_type_ids[env_ids] = torch.tensor(random_types, device=DEVICE)
    
    def _leader_state_batch(self, t):
        """批量计算领导者状态"""
        A = self.leader_amplitude
        omega = self.leader_omega
        phi = self.leader_phase
        
        pos = torch.zeros(self.num_envs, device=DEVICE)
        vel = torch.zeros(self.num_envs, device=DEVICE)
        
        # Sine
        sine_mask = self.trajectory_type_ids == 0
        if sine_mask.any():
            pos[sine_mask] = A[sine_mask] * torch.sin(omega[sine_mask] * t[sine_mask] + phi[sine_mask])
            vel[sine_mask] = A[sine_mask] * omega[sine_mask] * torch.cos(omega[sine_mask] * t[sine_mask] + phi[sine_mask])
        
        # Cosine
        cosine_mask = self.trajectory_type_ids == 1
        if cosine_mask.any():
            pos[cosine_mask] = A[cosine_mask] * torch.cos(omega[cosine_mask] * t[cosine_mask] + phi[cosine_mask])
            vel[cosine_mask] = -A[cosine_mask] * omega[cosine_mask] * torch.sin(omega[cosine_mask] * t[cosine_mask] + phi[cosine_mask])
        
        # Mixed
        mixed_mask = self.trajectory_type_ids == 2
        if mixed_mask.any():
            t_m, A_m, omega_m, phi_m = t[mixed_mask], A[mixed_mask], omega[mixed_mask], phi[mixed_mask]
            pos[mixed_mask] = A_m * (torch.sin(omega_m * t_m + phi_m) + 0.3 * torch.cos(0.5 * omega_m * t_m))
            vel[mixed_mask] = A_m * (omega_m * torch.cos(omega_m * t_m + phi_m) - 0.15 * omega_m * torch.sin(0.5 * omega_m * t_m))
        
        # Chirp
        chirp_mask = self.trajectory_type_ids == 3
        if chirp_mask.any():
            t_c, A_c, omega_c, phi_c = t[chirp_mask], A[chirp_mask], omega[chirp_mask], phi[chirp_mask]
            chirp_rate = 0.1
            inst_phase = (omega_c + chirp_rate * t_c) * t_c + phi_c
            inst_freq = omega_c + 2 * chirp_rate * t_c
            pos[chirp_mask] = A_c * torch.sin(inst_phase)
            vel[chirp_mask] = A_c * inst_freq * torch.cos(inst_phase)
        
        return pos, vel
    
    def reset(self, env_ids=None):
        """重置环境"""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=DEVICE)
            num_reset = self.num_envs
        else:
            num_reset = len(env_ids)
        
        self.t[env_ids] = 0.0
        
        if self.enable_randomization:
            self._randomize_leader_dynamics(env_ids)
        
        leader_pos, leader_vel = self._leader_state_batch(self.t)
        
        self.positions[env_ids, 0] = leader_pos[env_ids]
        self.velocities[env_ids, 0] = leader_vel[env_ids]
        
        if self.enable_randomization:
            pos_std = torch.rand(num_reset, 1, device=DEVICE) * \
                (self.pos_std_range[1] - self.pos_std_range[0]) + self.pos_std_range[0]
            vel_std = torch.rand(num_reset, 1, device=DEVICE) * \
                (self.vel_std_range[1] - self.vel_std_range[0]) + self.vel_std_range[0]
            
            self.positions[env_ids, 1:] = leader_pos[env_ids].unsqueeze(1) + \
                torch.randn(num_reset, self.num_followers, device=DEVICE) * pos_std
            self.velocities[env_ids, 1:] = leader_vel[env_ids].unsqueeze(1) + \
                torch.randn(num_reset, self.num_followers, device=DEVICE) * vel_std
        else:
            self.positions[env_ids, 1:] = leader_pos[env_ids].unsqueeze(1) + \
                torch.randn(num_reset, self.num_followers, device=DEVICE) * 0.3
            self.velocities[env_ids, 1:] = leader_vel[env_ids].unsqueeze(1) + \
                torch.randn(num_reset, self.num_followers, device=DEVICE) * 0.1
        
        self.positions[env_ids] = torch.clamp(self.positions[env_ids], -self.pos_limit, self.pos_limit)
        self.velocities[env_ids] = torch.clamp(self.velocities[env_ids], -self.vel_limit, self.vel_limit)
        
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
    
    def _compute_reward(self, tracking_error, comm_rate):
        """
        计算自适应奖励
        
        奖励 = 跟踪惩罚 + 改进奖励 + 通信惩罚
        
        核心设计：
        1. 跟踪惩罚（饱和型）: -tanh(error * scale) * max_penalty
           - 小误差时梯度大，大误差时饱和
           
        2. 通信惩罚（自适应权重）: -comm_rate * base * exp(-error * decay)
           - 误差大时权重低（专注跟踪）
           - 误差小时权重高（优化通信）
           
        3. 改进奖励: 鼓励误差持续减小
        """
        # ==================== 1. 跟踪惩罚（饱和型）====================
        # tracking_penalty ∈ [-max_penalty, 0]
        # 误差=0 时惩罚=0，误差→∞ 时惩罚→-max_penalty
        tracking_penalty = -torch.tanh(tracking_error * self.tracking_error_scale) * self.tracking_penalty_max
        
        # ==================== 2. 改进奖励 ====================
        improvement_bonus = torch.zeros_like(tracking_error)
        if self._prev_error is not None:
            # 误差减小 → 正奖励，误差增大 → 负奖励
            improvement = self._prev_error - tracking_error
            improvement_bonus = torch.clamp(
                improvement * self.improvement_scale, 
                -self.improvement_clip, 
                self.improvement_clip
            )
        self._prev_error = tracking_error.detach().clone()
        
        # ==================== 3. 通信惩罚（自适应权重）====================
        # comm_weight ∈ (0, 1]
        # 误差大时 → weight≈0（忽略通信成本）
        # 误差小时 → weight≈1（重视通信成本）
        comm_weight = torch.exp(-tracking_error * self.comm_weight_decay)
        
        # 有效通信惩罚
        comm_penalty = -comm_rate * self.comm_penalty_base * comm_weight
        
        # ==================== 总奖励 ====================
        raw_reward = tracking_penalty + improvement_bonus + comm_penalty
        
        return raw_reward, tracking_penalty, comm_penalty, comm_weight
    
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
        
        # 阈值映射
        normalized_threshold = (raw_threshold / self.th_scale).clamp(0.0, 1.0)
        threshold = self.threshold_min + (self.threshold_max - self.threshold_min) * normalized_threshold
        
        # 计算总控制
        base_u = self._compute_base_control()
        total_u = torch.clamp(base_u + delta_u, -20.0, 20.0)
        
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
        
        # ==================== 计算跟踪误差 ====================
        pos_error = torch.abs(self.positions[:, 1:] - self.positions[:, 0:1])
        vel_error = torch.abs(self.velocities[:, 1:] - self.velocities[:, 0:1])
        tracking_error = pos_error.mean(dim=1) + 0.5 * vel_error.mean(dim=1)
        
        # 通信率
        comm_rate = is_triggered.float().mean(dim=1)
        
        # ==================== 计算奖励 ====================
        raw_reward, tracking_penalty, comm_penalty, comm_weight = self._compute_reward(
            tracking_error, comm_rate
        )
        rewards = self._scale_reward_batch(raw_reward)
        
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)
        
        infos = {
            'tracking_error': tracking_error,
            'comm_rate': comm_rate,
            'leader_pos': self.positions[:, 0],
            'leader_vel': self.velocities[:, 0],
            'threshold_mean': threshold.mean(),
            # 奖励分解信息
            'tracking_penalty': tracking_penalty.mean(),
            'comm_penalty': comm_penalty.mean(),
            'comm_weight': comm_weight.mean(),
            # 领导者参数
            'leader_amplitude_mean': self.leader_amplitude.mean(),
            'leader_omega_mean': self.leader_omega.mean(),
        }
        
        return self._get_state(), rewards, dones, infos
    
    def get_leader_info(self):
        """获取领导者参数信息"""
        return {
            'amplitude': self.leader_amplitude.cpu().numpy(),
            'omega': self.leader_omega.cpu().numpy(),
            'phase': self.leader_phase.cpu().numpy(),
            'trajectory_types': [self.id_to_type[i.item()] for i in self.trajectory_type_ids]
        }


class LeaderFollowerMASEnv:
    """单环境版本"""
    
    def __init__(self, topology, enable_randomization=ENABLE_RANDOMIZATION):
        self.batched_env = BatchedLeaderFollowerEnv(
            topology, num_envs=1, enable_randomization=enable_randomization
        )
        self.topology = topology
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.role_ids = self.batched_env.role_ids
        self.enable_randomization = enable_randomization
    
    @property
    def positions(self):
        return self.batched_env.positions[0]
    
    @property
    def velocities(self):
        return self.batched_env.velocities[0]
    
    @property
    def t(self):
        return self.batched_env.t[0].item()
    
    @property
    def leader_amplitude(self):
        return self.batched_env.leader_amplitude[0].item()
    
    @property
    def leader_omega(self):
        return self.batched_env.leader_omega[0].item()
    
    @property
    def trajectory_type(self):
        type_id = self.batched_env.trajectory_type_ids[0].item()
        return self.batched_env.id_to_type[type_id]
    
    def reset(self):
        return self.batched_env.reset()[0]
    
    def step(self, action):
        states, rewards, dones, infos = self.batched_env.step(action.unsqueeze(0))
        info = {k: (v[0].item() if isinstance(v, torch.Tensor) and v.dim() > 0 else 
                    v.item() if isinstance(v, torch.Tensor) else v) 
                for k, v in infos.items()}
        return states[0], rewards[0].item(), dones[0].item(), info
    
    def get_leader_info(self):
        return {
            'amplitude': self.leader_amplitude,
            'omega': self.leader_omega,
            'phase': self.batched_env.leader_phase[0].item(),
            'trajectory_type': self.trajectory_type
        }