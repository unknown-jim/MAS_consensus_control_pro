"""训练脚本模块 - CTDE 架构。

本模块提供多智能体系统的训练和评估入口。

使用方法:
    python train.py
"""
import torch
import time
from typing import Optional, Tuple

from config import (
    NUM_FOLLOWERS, NUM_PINNED, MAX_STEPS, BATCH_SIZE,
    NUM_EPISODES, VIS_INTERVAL, SAVE_MODEL_PATH,
    print_config, set_seed, SEED,
    NUM_PARALLEL_ENVS, UPDATE_FREQUENCY, GRADIENT_STEPS,
    USE_AMP, DEVICE, WARMUP_STEPS
)
from topology import DirectedSpanningTreeTopology
from environment import BatchedLeaderFollowerEnv, LeaderFollowerMASEnv
from agent import SACAgent
from utils import collect_trajectory, plot_evaluation

try:
    from dashboard import TrainingDashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(num_episodes: int = NUM_EPISODES, vis_interval: int = VIS_INTERVAL,
          show_dashboard: bool = True,
          seed: int = SEED) -> Tuple[SACAgent, DirectedSpanningTreeTopology, Optional['TrainingDashboard']]:
    """CTDE 训练主函数。

    Args:
        num_episodes: 训练轮数。
        vis_interval: 可视化间隔。
        show_dashboard: 是否显示训练仪表盘。
        seed: 随机种子。

    Returns:
        agent: 训练好的智能体。
        topology: 通信拓扑。
        dashboard: 训练仪表盘（如果启用）。
    """
    set_seed(seed)
    print_config()

    reward_good_threshold = -0.33 * MAX_STEPS
    topology = DirectedSpanningTreeTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
    batched_env = BatchedLeaderFollowerEnv(topology, num_envs=NUM_PARALLEL_ENVS)
    eval_env = LeaderFollowerMASEnv(topology)
    agent = SACAgent(topology, use_amp=USE_AMP)

    dashboard = None
    if show_dashboard and HAS_DASHBOARD:
        dashboard = TrainingDashboard(num_episodes, vis_interval, max_steps=MAX_STEPS)
        dashboard.display()

    best_reward = -float('inf')
    global_step = 0
    start_time = time.time()
    warmup_printed = False

    print(f"\nCTDE Training Started")
    print(f"   Warmup Steps: {WARMUP_STEPS}")
    print(f"   Parallel envs: {NUM_PARALLEL_ENVS}")
    print(f"   Update Frequency: {UPDATE_FREQUENCY}\n")

    for episode in range(1, num_episodes + 1):
        states = batched_env.reset()
        episode_rewards = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_tracking_err = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_comm = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)

        for step in range(MAX_STEPS):
            global_step += NUM_PARALLEL_ENVS

            if not warmup_printed and agent.total_steps >= WARMUP_STEPS:
                print("Warmup complete! Starting policy updates...")
                warmup_printed = True

            if dashboard and step % 30 == 0:
                dashboard.update_step(step, MAX_STEPS)

            actions = agent.select_action(states, deterministic=False)
            next_states, rewards, dones, infos = batched_env.step(actions)

            agent.store_transitions_batch(states, actions, rewards, next_states, dones)

            if step % UPDATE_FREQUENCY == 0 and step > 0:
                agent.update(BATCH_SIZE, GRADIENT_STEPS)

            episode_rewards += rewards
            episode_tracking_err += infos['tracking_error']
            episode_comm += infos['comm_rate']
            states = next_states

        avg_reward = episode_rewards.mean().item()
        avg_tracking_err = (episode_tracking_err / MAX_STEPS).mean().item()
        avg_comm = (episode_comm / MAX_STEPS).mean().item()

        trajectory_data = None
        if episode % vis_interval == 0 or episode == 1 or avg_reward > best_reward:
            trajectory_data = collect_trajectory(agent, eval_env, MAX_STEPS)

        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(SAVE_MODEL_PATH)

        if dashboard:
            dashboard.update_episode(episode, avg_reward, avg_tracking_err, avg_comm,
                                     agent.last_losses, trajectory_data)
        elif episode % 20 == 0:
            elapsed = time.time() - start_time
            status = "Best" if avg_reward >= best_reward - 5 else (
                "Good" if avg_reward > reward_good_threshold * 1.5 else "Training")
            print(f"[{time.strftime('%H:%M:%S')}] {status} Ep {episode:4d} | "
                  f"R:{avg_reward:7.2f} | Err:{avg_tracking_err:.4f} | "
                  f"Comm:{avg_comm*100:.1f}% | alpha:{agent.alpha:.3f} | "
                  f"{episode/elapsed:.2f} ep/s")

    if dashboard:
        dashboard.finish()

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"CTDE Training Complete!")
    print(f"   Total Time: {elapsed/60:.1f} min")
    print(f"   Best Reward: {best_reward:.2f}")
    print(f"   Total Steps: {global_step:,}")
    print(f"{'='*60}")

    return agent, topology, dashboard


def evaluate(agent: SACAgent, topology: DirectedSpanningTreeTopology,
             num_tests: int = 5) -> dict:
    """评估模型。

    Args:
        agent: 智能体实例。
        topology: 拓扑结构。
        num_tests: 测试次数。

    Returns:
        评估结果字典。
    """
    from utils import evaluate_agent

    env = LeaderFollowerMASEnv(topology)
    results = evaluate_agent(agent, env, num_tests)

    print("\nEvaluation Results:")
    print(f"   Mean Reward: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"   Mean Tracking Error: {results['mean_tracking_error']:.4f}")
    print(f"   Mean Comm Rate: {results['mean_comm_rate']*100:.1f}%")

    return results


if __name__ == '__main__':
    agent, topology, _ = train(show_dashboard=False)
    evaluate(agent, topology, num_tests=5)
    plot_evaluation(agent, topology, num_tests=3)
