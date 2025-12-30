"""工具函数模块。

本模块提供训练和评估过程中使用的通用工具函数。

主要功能:
    - get_neighbor_obs: 获取邻居观测（供 agent 和 evaluate 共用）
    - collect_trajectory: 收集轨迹数据用于可视化
    - evaluate_agent: 评估智能体性能
    - plot_evaluation: 绘制评估结果图表
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple

from config import DEVICE, MAX_STEPS, STATE_DIM, MAX_NEIGHBORS, NUM_AGENTS, USE_NEIGHBOR_INFO


def get_neighbor_obs(states: torch.Tensor, neighbor_indices: Dict[int, List[int]],
                     num_agents: int = NUM_AGENTS,
                     max_neighbors: int = MAX_NEIGHBORS) -> Optional[torch.Tensor]:
    """获取邻居观测。

    为每个跟随者构建其邻居的观测张量，不足 max_neighbors 的部分用零填充。

    Args:
        states: 状态张量 (num_agents, state_dim)。
        neighbor_indices: 邻居索引字典，键为跟随者 ID，值为邻居 ID 列表。
        num_agents: 智能体总数。
        max_neighbors: 最大邻居数。

    Returns:
        邻居观测张量 (num_followers, max_neighbors, state_dim)，
        如果 USE_NEIGHBOR_INFO 为 False 则返回 None。
    """
    if not USE_NEIGHBOR_INFO:
        return None

    neighbor_obs_list = []
    for follower_id in range(1, num_agents):
        neighbors = neighbor_indices.get(follower_id, [])
        if len(neighbors) > 0:
            neighbor_states = states[neighbors]
            if len(neighbors) < max_neighbors:
                padding = torch.zeros(max_neighbors - len(neighbors), STATE_DIM, device=DEVICE)
                neighbor_states = torch.cat([neighbor_states, padding], dim=0)
        else:
            neighbor_states = torch.zeros(max_neighbors, STATE_DIM, device=DEVICE)
        neighbor_obs_list.append(neighbor_states)

    return torch.stack(neighbor_obs_list, dim=0)


@torch.no_grad()
def collect_trajectory(agent, env, max_steps: int = MAX_STEPS) -> Dict[str, np.ndarray]:
    """收集一条轨迹用于可视化。

    Args:
        agent: 智能体实例。
        env: 环境实例（单环境）。
        max_steps: 最大步数。

    Returns:
        包含以下键的字典:
            - times: 时间序列
            - leader_pos: 领导者位置序列
            - leader_vel: 领导者速度序列
            - follower_pos: 跟随者位置序列 (steps, num_followers)
            - follower_vel: 跟随者速度序列 (steps, num_followers)
    """
    state = env.reset()

    times = [0.0]
    leader_pos = [env.positions[0].item()]
    leader_vel = [env.velocities[0].item()]
    follower_pos = [env.positions[1:].cpu().numpy()]
    follower_vel = [env.velocities[1:].cpu().numpy()]

    for _ in range(max_steps):
        action = agent.select_action(state, deterministic=True)
        state, _, _, _ = env.step(action)

        times.append(env.t)
        leader_pos.append(env.positions[0].item())
        leader_vel.append(env.velocities[0].item())
        follower_pos.append(env.positions[1:].cpu().numpy())
        follower_vel.append(env.velocities[1:].cpu().numpy())

    return {
        'times': np.array(times),
        'leader_pos': np.array(leader_pos),
        'leader_vel': np.array(leader_vel),
        'follower_pos': np.array(follower_pos),
        'follower_vel': np.array(follower_vel)
    }


@torch.no_grad()
def evaluate_agent(agent, env, num_episodes: int = 5) -> Dict[str, float]:
    """评估智能体性能。

    Args:
        agent: 智能体实例。
        env: 环境实例（单环境）。
        num_episodes: 评估轮数。

    Returns:
        包含以下键的字典:
            - mean_reward: 平均奖励
            - std_reward: 奖励标准差
            - mean_tracking_error: 平均跟踪误差
            - mean_comm_rate: 平均通信率
    """
    results = {'rewards': [], 'tracking_errors': [], 'comm_rates': []}

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward, episode_tracking_err, episode_comm = 0, 0, 0

        for _ in range(MAX_STEPS):
            action = agent.select_action(state, deterministic=True)
            state, reward, _, info = env.step(action)

            episode_reward += reward
            episode_tracking_err += info['tracking_error']
            episode_comm += info['comm_rate']

        results['rewards'].append(episode_reward)
        results['tracking_errors'].append(episode_tracking_err / MAX_STEPS)
        results['comm_rates'].append(episode_comm / MAX_STEPS)

    return {
        'mean_reward': np.mean(results['rewards']),
        'std_reward': np.std(results['rewards']),
        'mean_tracking_error': np.mean(results['tracking_errors']),
        'mean_comm_rate': np.mean(results['comm_rates'])
    }


def plot_evaluation(agent, topology, num_tests: int = 3,
                    save_path: Optional[str] = None) -> List[Dict[str, float]]:
    """绘制评估结果。

    Args:
        agent: 智能体实例。
        topology: 拓扑结构。
        num_tests: 测试次数。
        save_path: 图片保存路径。

    Returns:
        每次测试的误差统计列表。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return []

    from environment import LeaderFollowerMASEnv

    env = LeaderFollowerMASEnv(topology)
    fig, axes = plt.subplots(num_tests, 2, figsize=(14, 4 * num_tests))
    if num_tests == 1:
        axes = axes.reshape(1, -1)

    results = []

    for test_idx in range(num_tests):
        traj = collect_trajectory(agent, env, MAX_STEPS)

        pos_errors = (traj['follower_pos'] - traj['leader_pos'][:, np.newaxis])**2
        final_error = np.mean(pos_errors[-1])
        avg_error = np.mean(pos_errors)
        results.append({'final_error': final_error, 'avg_error': avg_error})

        # 位置图
        ax1 = axes[test_idx, 0]
        ax1.plot(traj['times'], traj['leader_pos'], 'r-', linewidth=2.5, label='Leader')
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, traj['follower_pos'].shape[1]))
        for i in range(traj['follower_pos'].shape[1]):
            ax1.plot(traj['times'], traj['follower_pos'][:, i], color=colors[i],
                     alpha=0.7, linewidth=1.2, label=f'F{i+1}' if i < 3 else None)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position')
        ax1.set_title(f'Test {test_idx+1}: Position (Final Err: {final_error:.4f})')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 速度图
        ax2 = axes[test_idx, 1]
        ax2.plot(traj['times'], traj['leader_vel'], 'r-', linewidth=2.5, label='Leader')
        for i in range(traj['follower_vel'].shape[1]):
            ax2.plot(traj['times'], traj['follower_vel'][:, i], color=colors[i],
                     alpha=0.7, linewidth=1.2)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.set_title(f'Test {test_idx+1}: Velocity')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    plt.show()

    print("\nEvaluation Results:")
    print("-" * 40)
    for i, r in enumerate(results):
        print(f"Test {i+1}: Final Err = {r['final_error']:.4f}, Avg Err = {r['avg_error']:.4f}")

    return results
