"""领导者-跟随者多智能体系统环境模块。

本模块实现了基于事件触发通信的领导者-跟随者多智能体系统仿真环境，
采用自适应奖励设计来平衡跟踪性能和通信效率。

奖励设计核心思想:
    1. 跟踪惩罚（饱和型）: -tanh(error * scale) * max_penalty
       - 误差大时惩罚接近上限，避免过度惩罚
       - 误差小时惩罚近似线性，保持梯度
    2. 通信惩罚（自适应权重）: -comm_rate * base * exp(-error * decay)
       - 误差大时权重小，优先保证跟踪
       - 误差小时权重大，优化通信效率
    3. 改进奖励: 鼓励误差持续减小

环境类:
    - BatchedLeaderFollowerEnv: 向量化批量环境，支持并行仿真
    - LeaderFollowerMASEnv: 单环境封装，用于评估

Example:
    >>> from environment import BatchedLeaderFollowerEnv
    >>> from topology import DirectedSpanningTreeTopology
    >>> topology = DirectedSpanningTreeTopology(num_followers=6)
    >>> env = BatchedLeaderFollowerEnv(topology, num_envs=16)
    >>> state = env.reset()
    >>> next_state, reward, done, info = env.step(action)
"""
import torch
import numpy as np
from typing import Dict, Optional, Tuple

from config import (
    DEVICE, STATE_DIM, DT,
    THRESHOLD_MIN, THRESHOLD_MAX,
    LEADER_AMPLITUDE, LEADER_OMEGA, LEADER_PHASE,
    POS_LIMIT, VEL_LIMIT,
    REWARD_MIN, REWARD_MAX,
    TH_SCALE,
    TRACKING_ERROR_SCALE, TRACKING_PENALTY_MAX,
    COMM_PENALTY_BASE, COMM_WEIGHT_DECAY,
    IMPROVEMENT_SCALE, IMPROVEMENT_CLIP,
    ENABLE_RANDOMIZATION,
    LEADER_AMP_RANGE, LEADER_OMEGA_RANGE, LEADER_PHASE_RANGE,
    LEADER_TRAJECTORY_TYPES,
    FOLLOWER_POS_INIT_STD_RANGE, FOLLOWER_VEL_INIT_STD_RANGE
)


class BatchedLeaderFollowerEnv:
    """完全向量化的批量环境。

    支持多个环境实例的并行仿真，所有计算在 GPU 上向量化执行。

    特性:
        1. 领导者动力学随机化: 振幅、频率、相位、轨迹类型
        2. 跟随者初始状态随机化: 位置和速度的随机偏移
        3. 自适应奖励: 误差大时专注跟踪，误差小时优化通信
        4. 事件触发通信: 基于阈值的选择性状态广播

    Attributes:
        topology: 通信拓扑结构。
        num_envs: 并行环境数量。
        num_agents: 智能体总数。
        num_followers: 跟随者数量。
        enable_randomization: 是否启用随机化。
    """

    def __init__(self, topology, num_envs: int = 64,
                 enable_randomization: bool = ENABLE_RANDOMIZATION) -> None:
        """初始化批量环境。

        Args:
            topology: DirectedSpanningTreeTopology 实例。
            num_envs: 并行环境数量。
            enable_randomization: 是否启用领导者轨迹和跟随者状态随机化。
        """
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

        # 领导者参数（每个环境独立）
        self.leader_amplitude = torch.full((num_envs,), LEADER_AMPLITUDE, device=DEVICE)
        self.leader_omega = torch.full((num_envs,), LEADER_OMEGA, device=DEVICE)
        self.leader_phase = torch.full((num_envs,), LEADER_PHASE, device=DEVICE)

        # 轨迹类型映射
        self.trajectory_type_ids = torch.zeros(num_envs, dtype=torch.long, device=DEVICE)
        self.type_to_id = {'sine': 0, 'cosine': 1, 'mixed': 2, 'chirp': 3}
        self.id_to_type = {v: k for k, v in self.type_to_id.items()}

        # 环境物理参数
        self.pos_limit = POS_LIMIT
        self.vel_limit = VEL_LIMIT
        self.reward_min = REWARD_MIN
        self.reward_max = REWARD_MAX

        # 奖励函数参数
        self.tracking_error_scale = TRACKING_ERROR_SCALE
        self.tracking_penalty_max = TRACKING_PENALTY_MAX
        self.comm_penalty_base = COMM_PENALTY_BASE
        self.comm_weight_decay = COMM_WEIGHT_DECAY
        self.improvement_scale = IMPROVEMENT_SCALE
        self.improvement_clip = IMPROVEMENT_CLIP

        # 一致性控制器增益
        self.base_pos_gain = 5.0
        self.base_vel_gain = 2.5

        # 事件触发通信参数
        self.threshold_min = THRESHOLD_MIN
        self.threshold_max = THRESHOLD_MAX
        self.th_scale = TH_SCALE

        # 角色标识（0: 领导者, 1: 跟随者）
        self.role_ids = torch.zeros(self.num_agents, dtype=torch.long, device=DEVICE)
        self.role_ids[1:] = 1

        # 预计算邻居聚合矩阵
        self._precompute_neighbor_info()

        # 预分配状态张量
        self.positions = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.velocities = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_pos = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.last_broadcast_vel = torch.zeros(num_envs, self.num_agents, device=DEVICE)
        self.t = torch.zeros(num_envs, device=DEVICE)

        self._prev_error: Optional[torch.Tensor] = None
        self.reset()

    def _precompute_neighbor_info(self) -> None:
        """预计算邻居聚合矩阵。

        构建归一化邻接矩阵和 pinning 增益向量，
        用于高效计算一致性控制律。
        """
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

    def _randomize_leader_dynamics(self, env_ids: torch.Tensor) -> None:
        """随机化指定环境的领导者动力学参数。

        Args:
            env_ids: 需要随机化的环境索引。
        """
        if isinstance(env_ids, torch.Tensor):
            num_envs = len(env_ids)
        else:
            num_envs = self.num_envs
            env_ids = torch.arange(self.num_envs, device=DEVICE)

        # 随机化振幅、频率、相位
        self.leader_amplitude[env_ids] = torch.rand(num_envs, device=DEVICE) * \
            (self.amp_range[1] - self.amp_range[0]) + self.amp_range[0]

        self.leader_omega[env_ids] = torch.rand(num_envs, device=DEVICE) * \
            (self.omega_range[1] - self.omega_range[0]) + self.omega_range[0]

        self.leader_phase[env_ids] = torch.rand(num_envs, device=DEVICE) * \
            (self.phase_range[1] - self.phase_range[0]) + self.phase_range[0]

        # 随机化轨迹类型
        random_types = np.random.choice(
            [self.type_to_id[t] for t in self.trajectory_types],
            size=num_envs
        )
        self.trajectory_type_ids[env_ids] = torch.tensor(random_types, device=DEVICE)

    def _leader_state_batch(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量计算领导者状态。

        根据轨迹类型计算各环境领导者的位置和速度。

        Args:
            t: 时间向量，形状为 (num_envs,)。

        Returns:
            pos: 领导者位置，形状为 (num_envs,)。
            vel: 领导者速度，形状为 (num_envs,)。
        """
        A = self.leader_amplitude
        omega = self.leader_omega
        phi = self.leader_phase

        pos = torch.zeros(self.num_envs, device=DEVICE)
        vel = torch.zeros(self.num_envs, device=DEVICE)

        # 正弦轨迹
        sine_mask = self.trajectory_type_ids == 0
        if sine_mask.any():
            pos[sine_mask] = A[sine_mask] * torch.sin(
                omega[sine_mask] * t[sine_mask] + phi[sine_mask])
            vel[sine_mask] = A[sine_mask] * omega[sine_mask] * torch.cos(
                omega[sine_mask] * t[sine_mask] + phi[sine_mask])

        # 余弦轨迹
        cosine_mask = self.trajectory_type_ids == 1
        if cosine_mask.any():
            pos[cosine_mask] = A[cosine_mask] * torch.cos(
                omega[cosine_mask] * t[cosine_mask] + phi[cosine_mask])
            vel[cosine_mask] = -A[cosine_mask] * omega[cosine_mask] * torch.sin(
                omega[cosine_mask] * t[cosine_mask] + phi[cosine_mask])

        # 混合轨迹
        mixed_mask = self.trajectory_type_ids == 2
        if mixed_mask.any():
            t_m = t[mixed_mask]
            A_m, omega_m, phi_m = A[mixed_mask], omega[mixed_mask], phi[mixed_mask]
            pos[mixed_mask] = A_m * (torch.sin(omega_m * t_m + phi_m) +
                                     0.3 * torch.cos(0.5 * omega_m * t_m))
            vel[mixed_mask] = A_m * (omega_m * torch.cos(omega_m * t_m + phi_m) -
                                     0.15 * omega_m * torch.sin(0.5 * omega_m * t_m))

        # Chirp 轨迹（变频信号）
        chirp_mask = self.trajectory_type_ids == 3
        if chirp_mask.any():
            t_c = t[chirp_mask]
            A_c, omega_c, phi_c = A[chirp_mask], omega[chirp_mask], phi[chirp_mask]
            chirp_rate = 0.1
            inst_phase = (omega_c + chirp_rate * t_c) * t_c + phi_c
            inst_freq = omega_c + 2 * chirp_rate * t_c
            pos[chirp_mask] = A_c * torch.sin(inst_phase)
            vel[chirp_mask] = A_c * inst_freq * torch.cos(inst_phase)

        return pos, vel

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """重置环境。

        Args:
            env_ids: 需要重置的环境索引。如果为 None，则重置所有环境。

        Returns:
            初始状态，形状为 (num_envs, num_agents, state_dim)。
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=DEVICE)
            num_reset = self.num_envs
        else:
            num_reset = len(env_ids)

        self.t[env_ids] = 0.0

        if self.enable_randomization:
            self._randomize_leader_dynamics(env_ids)

        # 初始化领导者状态
        leader_pos, leader_vel = self._leader_state_batch(self.t)
        self.positions[env_ids, 0] = leader_pos[env_ids]
        self.velocities[env_ids, 0] = leader_vel[env_ids]

        # 初始化跟随者状态（在领导者附近随机分布）
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

        # 限制状态范围
        self.positions[env_ids] = torch.clamp(
            self.positions[env_ids], -self.pos_limit, self.pos_limit)
        self.velocities[env_ids] = torch.clamp(
            self.velocities[env_ids], -self.vel_limit, self.vel_limit)

        # 初始化广播状态
        self.last_broadcast_pos[env_ids] = self.positions[env_ids].clone()
        self.last_broadcast_vel[env_ids] = self.velocities[env_ids].clone()

        self._prev_error = None

        return self._get_state()

    def _get_state(self) -> torch.Tensor:
        """构建观测状态。

        状态包含:
            - 与邻居平均状态的误差（归一化）
            - 绝对位置和速度（归一化）

        Returns:
            状态张量，形状为 (num_envs, num_agents, state_dim)。
        """
        state = torch.zeros(self.num_envs, self.num_agents, STATE_DIM, device=DEVICE)

        # 计算邻居平均状态
        neighbor_avg_pos = torch.matmul(self.last_broadcast_pos, self.norm_adj_matrix.T)
        neighbor_avg_vel = torch.matmul(self.last_broadcast_vel, self.norm_adj_matrix.T)

        # 与邻居的误差
        pos_error = self.positions - neighbor_avg_pos
        vel_error = self.velocities - neighbor_avg_vel

        # 归一化状态
        state[:, :, 0] = pos_error / (self.pos_limit + 1e-6)
        state[:, :, 1] = vel_error / (self.vel_limit + 1e-6)
        state[:, :, 2] = self.positions / (self.pos_limit + 1e-6)
        state[:, :, 3] = self.velocities / (self.vel_limit + 1e-6)

        return state

    def _compute_base_control(self) -> torch.Tensor:
        """计算基础一致性控制律。

        基于广播状态计算分布式一致性控制输入。

        Returns:
            基础控制量，形状为 (num_envs, num_followers)。
        """
        follower_pos = self.last_broadcast_pos[:, 1:]
        follower_vel = self.last_broadcast_vel[:, 1:]

        leader_pos = self.last_broadcast_pos[:, 0:1]
        leader_vel = self.last_broadcast_vel[:, 0:1]

        # 跟随者间的邻接关系
        follower_adj = self.adj_matrix[1:, 1:]
        follower_degree = follower_adj.sum(dim=1, keepdim=True).clamp(min=1.0)

        # 一致性误差
        neighbor_pos_sum = torch.matmul(follower_pos, follower_adj.T)
        neighbor_vel_sum = torch.matmul(follower_vel, follower_adj.T)

        pos_consensus_error = follower_pos * follower_degree.T - neighbor_pos_sum
        vel_consensus_error = follower_vel * follower_degree.T - neighbor_vel_sum

        # Pinning 控制（与领导者的误差）
        pinning_gains_followers = self.pinning_gains[1:]
        pos_pinning_error = (follower_pos - leader_pos) * pinning_gains_followers
        vel_pinning_error = (follower_vel - leader_vel) * pinning_gains_followers

        # 组合控制律
        base_control = (
            -self.base_pos_gain * (pos_consensus_error + pos_pinning_error)
            - self.base_vel_gain * (vel_consensus_error + vel_pinning_error)
        )

        return base_control

    def _compute_reward(self, tracking_error: torch.Tensor,
                        comm_rate: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """计算自适应奖励。

        Args:
            tracking_error: 跟踪误差，形状为 (num_envs,)。
            comm_rate: 通信率，形状为 (num_envs,)。

        Returns:
            raw_reward: 原始奖励。
            tracking_penalty: 跟踪惩罚分量。
            comm_penalty: 通信惩罚分量。
            comm_weight: 通信惩罚权重。
        """
        # 跟踪惩罚（饱和型 tanh）
        tracking_penalty = -torch.tanh(
            tracking_error * self.tracking_error_scale) * self.tracking_penalty_max

        # 改进奖励（鼓励误差减小）
        improvement_bonus = torch.zeros_like(tracking_error)
        if self._prev_error is not None:
            improvement = self._prev_error - tracking_error
            improvement_bonus = torch.clamp(
                improvement * self.improvement_scale,
                -self.improvement_clip,
                self.improvement_clip
            )
        self._prev_error = tracking_error.detach().clone()

        # 通信惩罚（自适应权重：误差小时权重大）
        comm_weight = torch.exp(-tracking_error * self.comm_weight_decay)
        comm_penalty = -comm_rate * self.comm_penalty_base * comm_weight

        raw_reward = tracking_penalty + improvement_bonus + comm_penalty

        return raw_reward, tracking_penalty, comm_penalty, comm_weight

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor,
                                                   torch.Tensor, Dict]:
        """执行一步环境交互。

        Args:
            action: 动作张量，形状为 (num_envs, num_followers, action_dim)。
                action[:, :, 0]: 控制量修正 delta_u
                action[:, :, 1]: 事件触发阈值

        Returns:
            next_state: 下一状态，形状为 (num_envs, num_agents, state_dim)。
            rewards: 奖励，形状为 (num_envs,)。
            dones: 终止标志，形状为 (num_envs,)。
            infos: 包含调试信息的字典。
        """
        self.t += DT

        # 更新领导者状态
        leader_pos, leader_vel = self._leader_state_batch(self.t)
        self.positions[:, 0] = leader_pos
        self.velocities[:, 0] = leader_vel

        # 解析动作
        delta_u = action[:, :, 0] * 2.0
        raw_threshold = action[:, :, 1]

        # 阈值映射到有效范围
        normalized_threshold = (raw_threshold / self.th_scale).clamp(0.0, 1.0)
        threshold = self.threshold_min + \
            (self.threshold_max - self.threshold_min) * normalized_threshold

        # 计算总控制量
        base_u = self._compute_base_control()
        total_u = torch.clamp(base_u + delta_u, -20.0, 20.0)

        # 跟随者动力学（二阶积分器 + 非线性项）
        follower_pos = self.positions[:, 1:]
        follower_vel = self.velocities[:, 1:]

        nonlinear_term = 0.2 * torch.sin(follower_pos) - 0.1 * follower_vel
        acc = total_u + nonlinear_term

        new_vel = torch.clamp(follower_vel + acc * DT, -self.vel_limit, self.vel_limit)
        new_pos = torch.clamp(follower_pos + new_vel * DT, -self.pos_limit, self.pos_limit)

        self.positions[:, 1:] = new_pos
        self.velocities[:, 1:] = new_vel

        # 事件触发通信判断
        trigger_error = torch.abs(new_pos - self.last_broadcast_pos[:, 1:])
        is_triggered = trigger_error > threshold

        # 更新广播状态
        self.last_broadcast_pos[:, 1:] = torch.where(
            is_triggered, self.positions[:, 1:], self.last_broadcast_pos[:, 1:]
        )
        self.last_broadcast_vel[:, 1:] = torch.where(
            is_triggered, self.velocities[:, 1:], self.last_broadcast_vel[:, 1:]
        )
        self.last_broadcast_pos[:, 0] = self.positions[:, 0]
        self.last_broadcast_vel[:, 0] = self.velocities[:, 0]

        # 计算跟踪误差
        pos_error = torch.abs(self.positions[:, 1:] - self.positions[:, 0:1])
        vel_error = torch.abs(self.velocities[:, 1:] - self.velocities[:, 0:1])
        tracking_error = pos_error.mean(dim=1) + 0.5 * vel_error.mean(dim=1)

        # 计算通信率
        comm_rate = is_triggered.float().mean(dim=1)

        # 计算奖励
        raw_reward, tracking_penalty, comm_penalty, comm_weight = self._compute_reward(
            tracking_error, comm_rate
        )
        rewards = torch.clamp(raw_reward, self.reward_min, self.reward_max)

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=DEVICE)

        infos = {
            'tracking_error': tracking_error,
            'comm_rate': comm_rate,
            'leader_pos': self.positions[:, 0],
            'leader_vel': self.velocities[:, 0],
            'threshold_mean': threshold.mean(),
            'tracking_penalty': tracking_penalty.mean(),
            'comm_penalty': comm_penalty.mean(),
            'comm_weight': comm_weight.mean(),
            'leader_amplitude_mean': self.leader_amplitude.mean(),
            'leader_omega_mean': self.leader_omega.mean(),
        }

        return self._get_state(), rewards, dones, infos

    def get_leader_info(self) -> Dict:
        """获取领导者参数信息。

        Returns:
            包含振幅、频率、相位和轨迹类型的字典。
        """
        return {
            'amplitude': self.leader_amplitude.cpu().numpy(),
            'omega': self.leader_omega.cpu().numpy(),
            'phase': self.leader_phase.cpu().numpy(),
            'trajectory_types': [self.id_to_type[i.item()]
                                 for i in self.trajectory_type_ids]
        }


class LeaderFollowerMASEnv:
    """单环境封装。

    对 BatchedLeaderFollowerEnv 的单环境封装，
    提供更简洁的接口用于评估和可视化。

    Attributes:
        batched_env: 内部的批量环境实例（num_envs=1）。
        topology: 通信拓扑结构。
    """

    def __init__(self, topology,
                 enable_randomization: bool = ENABLE_RANDOMIZATION) -> None:
        """初始化单环境。

        Args:
            topology: DirectedSpanningTreeTopology 实例。
            enable_randomization: 是否启用随机化。
        """
        self.batched_env = BatchedLeaderFollowerEnv(
            topology, num_envs=1, enable_randomization=enable_randomization
        )
        self.topology = topology
        self.num_agents = topology.num_agents
        self.num_followers = topology.num_followers
        self.role_ids = self.batched_env.role_ids
        self.enable_randomization = enable_randomization

    @property
    def positions(self) -> torch.Tensor:
        """当前所有智能体的位置。"""
        return self.batched_env.positions[0]

    @property
    def velocities(self) -> torch.Tensor:
        """当前所有智能体的速度。"""
        return self.batched_env.velocities[0]

    @property
    def t(self) -> float:
        """当前仿真时间。"""
        return self.batched_env.t[0].item()

    @property
    def leader_amplitude(self) -> float:
        """领导者轨迹振幅。"""
        return self.batched_env.leader_amplitude[0].item()

    @property
    def leader_omega(self) -> float:
        """领导者轨迹角频率。"""
        return self.batched_env.leader_omega[0].item()

    @property
    def trajectory_type(self) -> str:
        """领导者轨迹类型。"""
        type_id = self.batched_env.trajectory_type_ids[0].item()
        return self.batched_env.id_to_type[type_id]

    def reset(self) -> torch.Tensor:
        """重置环境。

        Returns:
            初始状态，形状为 (num_agents, state_dim)。
        """
        return self.batched_env.reset()[0]

    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """执行一步环境交互。

        Args:
            action: 动作张量，形状为 (num_followers, action_dim)。

        Returns:
            next_state: 下一状态，形状为 (num_agents, state_dim)。
            reward: 标量奖励。
            done: 是否终止。
            info: 调试信息字典。
        """
        states, rewards, dones, infos = self.batched_env.step(action.unsqueeze(0))
        info = {k: (v[0].item() if isinstance(v, torch.Tensor) and v.dim() > 0 else
                    v.item() if isinstance(v, torch.Tensor) else v)
                for k, v in infos.items()}
        return states[0], rewards[0].item(), dones[0].item(), info

    def get_leader_info(self) -> Dict:
        """获取领导者参数信息。

        Returns:
            包含振幅、频率、相位和轨迹类型的字典。
        """
        return {
            'amplitude': self.leader_amplitude,
            'omega': self.leader_omega,
            'phase': self.batched_env.leader_phase[0].item(),
            'trajectory_type': self.trajectory_type
        }
