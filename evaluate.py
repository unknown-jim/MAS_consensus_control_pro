"""
评估训练好的模型
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from config import (
    DEVICE, NUM_FOLLOWERS, MAX_STEPS, SAVE_MODEL_PATH,
    STATE_DIM, HIDDEN_DIM, ACTION_DIM, NUM_AGENTS,
    USE_NEIGHBOR_INFO, MAX_NEIGHBORS, set_seed, SEED
)
from topology import DirectedSpanningTreeTopology
from environment import LeaderFollowerMASEnv
from networks import DecentralizedActor


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path=SAVE_MODEL_PATH, use_fixed_seed=True, enable_randomization=False):
        """
        Args:
            model_path: 模型路径
            use_fixed_seed: 是否使用固定种子
            enable_randomization: 是否启用领导者轨迹随机化
                - False: 使用固定轨迹（与训练中 collect_trajectory 评估一致）
                - True: 随机轨迹（测试泛化性）
        """
        self.model_path = model_path
        self.use_fixed_seed = use_fixed_seed
        self.enable_randomization = enable_randomization
        
        # 固定种子以复现训练时的环境
        if use_fixed_seed:
            set_seed(SEED)
            print(f"🎲 使用固定随机种子: {SEED}")
        
        self.topology = DirectedSpanningTreeTopology(NUM_FOLLOWERS)
        # 关键：enable_randomization 控制领导者轨迹是否随机化
        self.env = LeaderFollowerMASEnv(self.topology, enable_randomization=enable_randomization)
        
        if enable_randomization:
            print(f"🎲 领导者轨迹: 随机化 (测试泛化性)")
        else:
            print(f"📌 领导者轨迹: 固定 (复现训练评估环境)")
        
        # 加载模型
        self.actor = self._load_model()
        
        # 预计算邻居信息
        if USE_NEIGHBOR_INFO:
            self._precompute_neighbor_info()
    
    def _load_model(self):
        """加载训练好的模型"""
        actor = DecentralizedActor(
            STATE_DIM, HIDDEN_DIM,
            use_neighbor_info=USE_NEIGHBOR_INFO
        ).to(DEVICE)
        
        checkpoint = torch.load(self.model_path, map_location=DEVICE)
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()
        
        print(f"✅ 模型加载成功: {self.model_path}")
        print(f"   训练 Episode: {checkpoint.get('episode', 'N/A')}")
        reward = checkpoint.get('reward', 'N/A')
        if isinstance(reward, (int, float)):
            print(f"   最佳奖励: {reward:.2f}")
        else:
            print(f"   最佳奖励: {reward}")
        
        return actor
    
    def _precompute_neighbor_info(self):
        """预计算邻居索引"""
        self.neighbor_indices = {}
        for follower_id in range(1, NUM_AGENTS):
            neighbors = self.topology.get_neighbors(follower_id)
            self.neighbor_indices[follower_id] = neighbors[:MAX_NEIGHBORS]
    
    def _get_neighbor_obs(self, states):
        """获取邻居观测"""
        if not USE_NEIGHBOR_INFO:
            return None
        
        neighbor_obs_list = []
        for follower_id in range(1, NUM_AGENTS):
            neighbors = self.neighbor_indices.get(follower_id, [])
            if len(neighbors) > 0:
                neighbor_states = states[neighbors]
                if len(neighbors) < MAX_NEIGHBORS:
                    padding = torch.zeros(MAX_NEIGHBORS - len(neighbors), STATE_DIM, device=DEVICE)
                    neighbor_states = torch.cat([neighbor_states, padding], dim=0)
            else:
                neighbor_states = torch.zeros(MAX_NEIGHBORS, STATE_DIM, device=DEVICE)
            neighbor_obs_list.append(neighbor_states)
        
        return torch.stack(neighbor_obs_list, dim=0)
    
    @torch.no_grad()
    def select_action(self, states, deterministic=True):
        """选择动作"""
        follower_states = states[1:]  # 排除领导者
        neighbor_obs = self._get_neighbor_obs(states)
        
        if deterministic:
            actions, _ = self.actor(follower_states, neighbor_obs, deterministic=True)
        else:
            actions, _ = self.actor(follower_states, neighbor_obs, deterministic=False)
        
        return actions
    
    def run_episode(self, deterministic=True, render=False, seed=None):
        """运行一个 episode"""
        # 可选：为每个 episode 设置不同种子
        if seed is not None:
            set_seed(seed)
        
        state = self.env.reset()
        
        # 记录数据
        rewards = []
        errors = []
        comm_rates = []
        leader_positions = []
        follower_positions = []
        leader_velocities = []
        follower_velocities = []
        
        for step in range(MAX_STEPS):
            action = self.select_action(state, deterministic=deterministic)
            next_state, reward, done, info = self.env.step(action)
            
            # 处理 reward 可能是 tensor 或 float
            if hasattr(reward, 'mean'):
                rewards.append(reward.mean().item())
            else:
                rewards.append(float(reward))
            errors.append(info['tracking_error'])
            comm_rates.append(info['comm_rate'])
            
            leader_positions.append(self.env.positions[0].item())
            follower_positions.append(self.env.positions[1:].cpu().numpy())
            leader_velocities.append(self.env.velocities[0].item())
            follower_velocities.append(self.env.velocities[1:].cpu().numpy())
            
            state = next_state
            
            if done:
                break
        
        return {
            'total_reward': sum(rewards),
            'avg_error': np.mean(errors),
            'final_error': errors[-1],
            'avg_comm_rate': np.mean(comm_rates),
            'leader_pos': np.array(leader_positions),
            'follower_pos': np.array(follower_positions),
            'leader_vel': np.array(leader_velocities),
            'follower_vel': np.array(follower_velocities),
            'errors': np.array(errors),
            'comm_rates': np.array(comm_rates)
        }
    
    def evaluate(self, num_episodes=10, deterministic=True, use_different_seeds=False):
        """评估模型性能
        
        Args:
            num_episodes: 评估的 episode 数量
            deterministic: 是否使用确定性策略
            use_different_seeds: 是否为每个 episode 使用不同种子
        """
        print(f"\n{'='*60}")
        print(f"🔍 模型评估 (共 {num_episodes} 个 episodes)")
        if self.enable_randomization:
            print(f"   领导者轨迹: 随机化")
        else:
            print(f"   领导者轨迹: 固定")
        if use_different_seeds:
            print(f"   随机种子: 不同")
        else:
            print(f"   随机种子: 固定 (SEED={SEED})")
        print(f"{'='*60}")
        
        all_rewards = []
        all_errors = []
        all_comm_rates = []
        
        for ep in range(num_episodes):
            # 为每个 episode 设置种子
            if use_different_seeds:
                seed = SEED + ep  # 不同种子测试泛化
            else:
                seed = SEED  # 固定种子复现
            
            set_seed(seed)
            result = self.run_episode(deterministic=deterministic)
            all_rewards.append(result['total_reward'])
            all_errors.append(result['avg_error'])
            all_comm_rates.append(result['avg_comm_rate'])
            
            print(f"  Episode {ep+1:2d} | R: {result['total_reward']:7.1f} | "
                  f"Err: {result['avg_error']:.4f} | Comm: {result['avg_comm_rate']*100:.1f}%")
        
        print(f"\n{'='*60}")
        print(f"📊 评估结果汇总:")
        print(f"{'='*60}")
        print(f"  平均奖励: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
        print(f"  平均误差: {np.mean(all_errors):.4f} ± {np.std(all_errors):.4f}")
        print(f"  平均通信率: {np.mean(all_comm_rates)*100:.1f}% ± {np.std(all_comm_rates)*100:.1f}%")
        print(f"  最佳奖励: {max(all_rewards):.2f}")
        print(f"  最小误差: {min(all_errors):.4f}")
        print(f"{'='*60}")
        
        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_error': np.mean(all_errors),
            'std_error': np.std(all_errors),
            'mean_comm': np.mean(all_comm_rates),
            'std_comm': np.std(all_comm_rates)
        }
    
    def visualize(self, save_path='evaluation_result.png'):
        """可视化一个 episode 的结果"""
        result = self.run_episode(deterministic=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Model Evaluation (R={result["total_reward"]:.1f}, Err={result["avg_error"]:.4f})', 
                     fontsize=14, fontweight='bold')
        
        time = np.arange(len(result['leader_pos'])) * 0.05  # DT=0.05
        
        # 1. 位置跟踪
        ax1 = axes[0, 0]
        ax1.plot(time, result['leader_pos'], 'r-', linewidth=2, label='Leader')
        follower_pos = result['follower_pos']
        for i in range(follower_pos.shape[1]):
            ax1.plot(time, follower_pos[:, i], '--', alpha=0.5, linewidth=1)
        ax1.plot(time, follower_pos.mean(axis=1), 'b-', linewidth=2, label='Avg Follower')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position')
        ax1.set_title('Position Tracking')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 速度跟踪
        ax2 = axes[0, 1]
        ax2.plot(time, result['leader_vel'], 'r-', linewidth=2, label='Leader')
        follower_vel = result['follower_vel']
        ax2.plot(time, follower_vel.mean(axis=1), 'b-', linewidth=2, label='Avg Follower')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity Tracking')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 跟踪误差
        ax3 = axes[1, 0]
        ax3.plot(time, result['errors'], 'g-', linewidth=1.5)
        ax3.axhline(y=result['avg_error'], color='r', linestyle='--', label=f'Avg: {result["avg_error"]:.4f}')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Tracking Error')
        ax3.set_title('Tracking Error over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 通信率
        ax4 = axes[1, 1]
        ax4.plot(time, result['comm_rates'] * 100, 'purple', linewidth=1.5)
        ax4.axhline(y=result['avg_comm_rate']*100, color='r', linestyle='--', 
                    label=f'Avg: {result["avg_comm_rate"]*100:.1f}%')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Communication Rate (%)')
        ax4.set_title('Communication Rate over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"\n📈 可视化结果已保存: {save_path}")
        
        return result


def main():
    """主函数"""
    # 检查模型文件是否存在
    if not Path(SAVE_MODEL_PATH).exists():
        print(f"❌ 模型文件不存在: {SAVE_MODEL_PATH}")
        return
    
    # ============================================================
    # 1. 固定轨迹评估（复现训练中的评估环境）
    # ============================================================
    print("\n" + "="*60)
    print("📌 固定轨迹评估 (复现训练评估环境)")
    print("   - 领导者轨迹: 固定 (sine, A=2.0, ω=0.5)")
    print("   - 随机种子: 固定")
    print("="*60)
    
    evaluator_fixed = ModelEvaluator(
        use_fixed_seed=True, 
        enable_randomization=False  # 关闭随机化
    )
    stats_fixed = evaluator_fixed.evaluate(num_episodes=5, deterministic=True, use_different_seeds=False)
    
    # 可视化固定轨迹结果
    set_seed(SEED)
    evaluator_fixed.visualize(save_path='evaluation_fixed.png')
    
    # ============================================================
    # 2. 随机轨迹评估（测试泛化性）
    # ============================================================
    print("\n" + "="*60)
    print("🎲 随机轨迹评估 (测试泛化性)")
    print("   - 领导者轨迹: 随机化")
    print("   - 随机种子: 不同")
    print("="*60)
    
    evaluator_random = ModelEvaluator(
        use_fixed_seed=False,
        enable_randomization=True  # 开启随机化
    )
    stats_random = evaluator_random.evaluate(num_episodes=5, deterministic=True, use_different_seeds=True)
    
    # 可视化随机轨迹结果
    evaluator_random.visualize(save_path='evaluation_random.png')
    
    # ============================================================
    # 对比总结
    # ============================================================
    print("\n" + "="*60)
    print("📊 评估对比总结")
    print("="*60)
    print(f"  固定轨迹: R={stats_fixed['mean_reward']:.2f}±{stats_fixed['std_reward']:.2f}, "
          f"Err={stats_fixed['mean_error']:.4f}")
    print(f"  随机轨迹: R={stats_random['mean_reward']:.2f}±{stats_random['std_reward']:.2f}, "
          f"Err={stats_random['mean_error']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
