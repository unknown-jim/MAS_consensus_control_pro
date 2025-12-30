"""经验回放缓冲区模块。

本模块实现了 GPU 优化的经验回放缓冲区，用于存储和采样
强化学习训练过程中的状态转移经验。

特点:
    - GPU 内存预分配，避免频繁的内存分配
    - 支持单条和批量经验存储
    - 环形缓冲区实现，自动覆盖旧数据
    - 高效的随机采样

Example:
    >>> from buffer import OptimizedReplayBuffer
    >>> buffer = OptimizedReplayBuffer(capacity=100000, num_agents=7)
    >>> buffer.push_batch(states, actions, rewards, next_states, dones)
    >>> if buffer.is_ready(batch_size=256):
    ...     batch = buffer.sample(batch_size=256)
"""
import torch
from typing import Tuple

from config import DEVICE, BUFFER_SIZE, STATE_DIM, ACTION_DIM, NUM_AGENTS


class OptimizedReplayBuffer:
    """GPU 预分配的高效经验回放缓冲区。

    使用环形缓冲区结构，在 GPU 上预分配固定大小的内存，
    避免训练过程中的动态内存分配开销。

    Attributes:
        capacity: 缓冲区最大容量。
        num_agents: 智能体总数。
        num_followers: 跟随者数量。
        state_dim: 状态维度。
        action_dim: 动作维度。
        ptr: 当前写入位置指针。
        size: 当前存储的经验数量。
    """

    def __init__(self, capacity: int = BUFFER_SIZE, num_agents: int = NUM_AGENTS,
                 state_dim: int = STATE_DIM, action_dim: int = ACTION_DIM) -> None:
        """初始化经验回放缓冲区。

        在 GPU 上预分配所有存储张量。

        Args:
            capacity: 缓冲区最大容量。
            num_agents: 智能体总数（包括领导者）。
            state_dim: 单个智能体的状态维度。
            action_dim: 单个智能体的动作维度。
        """
        self.capacity = capacity
        self.num_agents = num_agents
        self.num_followers = num_agents - 1
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.ptr = 0
        self.size = 0

        # 预分配 GPU 内存
        self.states = torch.zeros(capacity, num_agents, state_dim, device=DEVICE)
        self.actions = torch.zeros(capacity, self.num_followers, action_dim, device=DEVICE)
        self.rewards = torch.zeros(capacity, device=DEVICE)
        self.next_states = torch.zeros(capacity, num_agents, state_dim, device=DEVICE)
        self.dones = torch.zeros(capacity, device=DEVICE)

    def push(self, state: torch.Tensor, action: torch.Tensor, reward: float,
             next_state: torch.Tensor, done: bool) -> None:
        """存储单条经验。

        Args:
            state: 当前状态，形状为 (num_agents, state_dim)。
            action: 执行的动作，形状为 (num_followers, action_dim)。
            reward: 获得的奖励（标量）。
            next_state: 下一状态，形状为 (num_agents, state_dim)。
            done: 是否为终止状态。
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def push_batch(self, states: torch.Tensor, actions: torch.Tensor,
                   rewards: torch.Tensor, next_states: torch.Tensor,
                   dones: torch.Tensor) -> None:
        """批量存储经验。

        高效地存储一批经验，自动处理缓冲区边界的环绕情况。

        Args:
            states: 当前状态批次，形状为 (batch, num_agents, state_dim)。
            actions: 动作批次，形状为 (batch, num_followers, action_dim)。
            rewards: 奖励批次，形状为 (batch,)。
            next_states: 下一状态批次，形状为 (batch, num_agents, state_dim)。
            dones: 终止标志批次，形状为 (batch,)。
        """
        batch_size = states.shape[0]

        if self.ptr + batch_size <= self.capacity:
            # 不需要环绕，直接写入
            idx = slice(self.ptr, self.ptr + batch_size)
            self.states[idx] = states
            self.actions[idx] = actions
            self.rewards[idx] = rewards
            self.next_states[idx] = next_states
            self.dones[idx] = dones.float()
        else:
            # 需要环绕，分两部分写入
            first_part = self.capacity - self.ptr
            second_part = batch_size - first_part

            self.states[self.ptr:] = states[:first_part]
            self.states[:second_part] = states[first_part:]

            self.actions[self.ptr:] = actions[:first_part]
            self.actions[:second_part] = actions[first_part:]

            self.rewards[self.ptr:] = rewards[:first_part]
            self.rewards[:second_part] = rewards[first_part:]

            self.next_states[self.ptr:] = next_states[:first_part]
            self.next_states[:second_part] = next_states[first_part:]

            self.dones[self.ptr:] = dones[:first_part].float()
            self.dones[:second_part] = dones[first_part:].float()

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """随机采样一批经验。

        Args:
            batch_size: 采样数量。

        Returns:
            包含以下元素的元组:
                - states: 形状为 (batch, num_agents, state_dim)
                - actions: 形状为 (batch, num_followers, action_dim)
                - rewards: 形状为 (batch,)
                - next_states: 形状为 (batch, num_agents, state_dim)
                - dones: 形状为 (batch,)
        """
        indices = torch.randint(0, self.size, (batch_size,), device=DEVICE)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices]
        )

    def __len__(self) -> int:
        """返回当前存储的经验数量。"""
        return self.size

    def is_ready(self, batch_size: int) -> bool:
        """检查缓冲区是否有足够的经验进行采样。

        Args:
            batch_size: 需要的采样数量。

        Returns:
            如果存储的经验数量 >= batch_size 则返回 True。
        """
        return self.size >= batch_size
