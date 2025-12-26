"""
训练脚本 - 课程学习优化版
"""
import torch
import time

from config import (
    NUM_FOLLOWERS, NUM_PINNED, MAX_STEPS, BATCH_SIZE,
    NUM_EPISODES, VIS_INTERVAL, SAVE_MODEL_PATH, 
    print_config, set_seed, SEED,
    NUM_PARALLEL_ENVS, UPDATE_FREQUENCY, GRADIENT_STEPS,
    USE_AMP, DEVICE,
    get_comm_penalty, get_threshold_bounds, get_comm_bonus, get_curriculum_progress,
    COMM_PENALTY_WARMUP, COMM_PENALTY_ANNEAL
)
from topology import DirectedSpanningTreeTopology
from environment import BatchedLeaderFollowerEnv, LeaderFollowerMASEnv
from agent import SACAgent
from utils import collect_trajectory, plot_evaluation

# 可选导入 dashboard
try:
    from dashboard import TrainingDashboard
    HAS_DASHBOARD = True
except ImportError:
    HAS_DASHBOARD = False
    print("⚠️ Dashboard not available, using console logging")


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def train(num_episodes=NUM_EPISODES, vis_interval=VIS_INTERVAL, 
          show_dashboard=True, seed=SEED):
    """课程学习优化训练"""
    set_seed(seed)
    print_config()
    
    # 初始化
    topology = DirectedSpanningTreeTopology(NUM_FOLLOWERS, num_pinned=NUM_PINNED)
    batched_env = BatchedLeaderFollowerEnv(topology, num_envs=NUM_PARALLEL_ENVS)
    eval_env = LeaderFollowerMASEnv(topology)
    
    agent = SACAgent(topology, use_amp=USE_AMP)
    
    dashboard = None
    if show_dashboard and HAS_DASHBOARD:
        dashboard = TrainingDashboard(num_episodes, vis_interval)
        dashboard.display()
    
    best_reward = -float('inf')
    global_step = 0
    
    start_time = time.time()
    log_interval = 10
    
    # 🔧 打印课程学习阶段信息
    print("\n📚 Curriculum Learning Schedule:")
    print(f"   Phase 1 (WARMUP):  Ep 1-{COMM_PENALTY_WARMUP}")
    print(f"   Phase 2 (ANNEAL):  Ep {COMM_PENALTY_WARMUP+1}-{COMM_PENALTY_WARMUP+COMM_PENALTY_ANNEAL}")
    print(f"   Phase 3 (FULL):    Ep {COMM_PENALTY_WARMUP+COMM_PENALTY_ANNEAL+1}-{num_episodes}")
    print()
    
    # 训练循环
    for episode in range(1, num_episodes + 1):
        
        # 🔧 课程学习：获取当前阶段的所有参数
        current_comm_penalty = get_comm_penalty(episode)
        threshold_min, threshold_max = get_threshold_bounds(episode)
        current_comm_bonus = get_comm_bonus(episode)
        current_progress = get_curriculum_progress(episode)
        
        # 🔧 设置环境的课程学习参数
        batched_env.set_curriculum_params(
            comm_penalty=current_comm_penalty,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            comm_bonus=current_comm_bonus,
            progress=current_progress
        )
        eval_env.set_curriculum_params(
            comm_penalty=current_comm_penalty,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            comm_bonus=current_comm_bonus,
            progress=current_progress
        )
        
        states = batched_env.reset()
        
        episode_rewards = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_tracking_err = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        episode_comm = torch.zeros(NUM_PARALLEL_ENVS, device=DEVICE)
        
        for step in range(MAX_STEPS):
            global_step += NUM_PARALLEL_ENVS
            
            # 更新步进度
            if dashboard and step % 10 == 0:
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
        
        # 可视化
        trajectory_data = None
        if episode % vis_interval == 0 or episode == 1:
            trajectory_data = collect_trajectory(agent, eval_env, MAX_STEPS)
        
        if avg_reward > best_reward:
            best_reward = avg_reward
            agent.save(SAVE_MODEL_PATH)
            trajectory_data = collect_trajectory(agent, eval_env, MAX_STEPS)
        
        # 🔧 添加课程学习信息到 losses
        agent.last_losses['comm_penalty'] = current_comm_penalty
        agent.last_losses['comm_bonus'] = current_comm_bonus
        agent.last_losses['threshold_range'] = (threshold_min, threshold_max)
        agent.last_losses['curriculum_progress'] = current_progress
        
        if dashboard:
            dashboard.update_episode(
                episode, avg_reward, avg_tracking_err, avg_comm,
                agent.last_losses, trajectory_data
            )
        elif episode % log_interval == 0:
            elapsed = time.time() - start_time
            speed = episode / elapsed
            
            # 🔧 显示课程学习阶段
            if current_progress < 0.01:
                phase = "🎓 WARMUP"
            elif current_progress < 0.99:
                phase = f"📈 ANNEAL ({current_progress*100:.0f}%)"
            else:
                phase = "🎯 FULL"
            
            print(f"Ep {episode:4d} | R:{avg_reward:7.2f} | Err:{avg_tracking_err:.4f} | "
                  f"Comm:{avg_comm*100:.1f}% | Th:[{threshold_min:.3f},{threshold_max:.3f}] | "
                  f"{phase} | {speed:.2f} ep/s")
    
    if dashboard:
        dashboard.finish()
    
    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"✅ Training Complete!")
    print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"   Speed: {num_episodes/elapsed:.2f} ep/s")
    print(f"   Total steps: {global_step:,}")
    print(f"   Best reward: {best_reward:.2f}")
    print(f"{'='*60}")
    
    return agent, topology, dashboard


if __name__ == '__main__':
    agent, topology, _ = train(show_dashboard=False)
    plot_evaluation(agent, topology, num_tests=3, save_path='evaluation.png')