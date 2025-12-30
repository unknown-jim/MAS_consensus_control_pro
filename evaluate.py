"""
评估训练好的模型
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional

from config import (
    DEVICE, NUM_FOLLOWERS, MAX_STEPS, SAVE_MODEL_PATH,
    STATE_DIM, HIDDEN_DIM, NUM_AGENTS,
    USE_NEIGHBOR_INFO, MAX_NEIGHBORS, set_seed, SEED
)
from topology import DirectedSpanningTreeTopology
from environment import LeaderFollowerMASEnv
from networks import DecentralizedActor
from utils import get_neighbor_obs


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, model_path: str = SAVE_MODEL_PATH, use_fixed_seed: bool = True, 
                 enable_randomization: bool = False):
        """
        Args:
            model_path: 模型路径
            use_fixed_seed: 是否使用固定种子
            enable_randomization: 是否启用领导者轨迹随机化
        """
        self.model_path = model_path
        self.use_fixed_seed = use_fixed_seed
        self.enable_randomization = enable_randomization
        
        if use_fixed_seed:
            set_seed(SEED)
            print(f"Using fixed seed: {SEED}")
        
        self.topology = DirectedSpanningTreeTopology(NUM_FOLLOWERS)
        self.env = LeaderFollowerMASEnv(self.topology, enable_randomization=enable_randomization)
        
        if enable_randomization:
            print(f"Leader trajectory: Randomized (testing generalization)")
        else:
            print(f"Leader trajectory: Fixed (reproducing training evaluation)")
        
        self.actor = self._load_model()
        
        if USE_NEIGHBOR_INFO:
            self._precompute_neighbor_info()
    
    def _load_model(self) -> DecentralizedActor:
        """加载训练好的模型"""
        actor = DecentralizedActor(
            STATE_DIM, HIDDEN_DIM,
            use_neighbor_info=USE_NEIGHBOR_INFO
        ).to(DEVICE)
        
        checkpoint = torch.load(self.model_path, map_location=DEVICE, weights_only=False)
        actor.load_state_dict(checkpoint['actor'])
        actor.eval()
        
        print(f"Model loaded: {self.model_path}")
        print(f"   Training Episode: {checkpoint.get('episode', 'N/A')}")
        reward = checkpoint.get('reward', 'N/A')
        if isinstance(reward, (int, float)):
            print(f"   Best Reward: {reward:.2f}")
        else:
            print(f"   Best Reward: {reward}")
        
        return actor
    
    def _precompute_neighbor_info(self) -> None:
        """预计算邻居索引"""
        self.neighbor_indices = {}
        for follower_id in range(1, NUM_AGENTS):
            neighbors = self.topology.get_neighbors(follower_id)
            self.neighbor_indices[follower_id] = neighbors[:MAX_NEIGHBORS]
    
    @torch.no_grad()
    def select_action(self, states: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """选择动作"""
        follower_states = states[1:]
        neighbor_obs = get_neighbor_obs(states, self.neighbor_indices) if USE_NEIGHBOR_INFO else None
        
        actions, _ = self.actor(follower_states, neighbor_obs, deterministic=deterministic)
        return actions
    
    def run_episode(self, deterministic: bool = True, seed: Optional[int] = None) -> Dict:
        """运行一个 episode"""
        if seed is not None:
            set_seed(seed)
        
        state = self.env.reset()
        
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
    
    def evaluate(self, num_episodes: int = 10, deterministic: bool = True, 
                 use_different_seeds: bool = False) -> Dict:
        """评估模型性能"""
        print(f"\n{'='*60}")
        print(f"Model Evaluation ({num_episodes} episodes)")
        if self.enable_randomization:
            print(f"   Leader trajectory: Randomized")
        else:
            print(f"   Leader trajectory: Fixed")
        if use_different_seeds:
            print(f"   Random seed: Different")
        else:
            print(f"   Random seed: Fixed (SEED={SEED})")
        print(f"{'='*60}")
        
        all_rewards = []
        all_errors = []
        all_comm_rates = []
        
        for ep in range(num_episodes):
            if use_different_seeds:
                seed = SEED + ep
            else:
                seed = SEED
            
            set_seed(seed)
            result = self.run_episode(deterministic=deterministic)
            all_rewards.append(result['total_reward'])
            all_errors.append(result['avg_error'])
            all_comm_rates.append(result['avg_comm_rate'])
            
            print(f"  Episode {ep+1:2d} | R: {result['total_reward']:7.1f} | "
                  f"Err: {result['avg_error']:.4f} | Comm: {result['avg_comm_rate']*100:.1f}%")
        
        print(f"\n{'='*60}")
        print(f"Evaluation Summary:")
        print(f"{'='*60}")
        print(f"  Mean Reward: {np.mean(all_rewards):.2f} +/- {np.std(all_rewards):.2f}")
        print(f"  Mean Error: {np.mean(all_errors):.4f} +/- {np.std(all_errors):.4f}")
        print(f"  Mean Comm Rate: {np.mean(all_comm_rates)*100:.1f}% +/- {np.std(all_comm_rates)*100:.1f}%")
        print(f"  Best Reward: {max(all_rewards):.2f}")
        print(f"  Min Error: {min(all_errors):.4f}")
        print(f"{'='*60}")
        
        return {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_error': np.mean(all_errors),
            'std_error': np.std(all_errors),
            'mean_comm': np.mean(all_comm_rates),
            'std_comm': np.std(all_comm_rates)
        }
    
    def visualize(self, save_path: str = 'evaluation_result.png') -> Dict:
        """可视化一个 episode 的结果"""
        result = self.run_episode(deterministic=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Model Evaluation (R={result["total_reward"]:.1f}, Err={result["avg_error"]:.4f})', 
                     fontsize=14, fontweight='bold')
        
        time = np.arange(len(result['leader_pos'])) * 0.05
        
        # Position tracking
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
        
        # Velocity tracking
        ax2 = axes[0, 1]
        ax2.plot(time, result['leader_vel'], 'r-', linewidth=2, label='Leader')
        follower_vel = result['follower_vel']
        ax2.plot(time, follower_vel.mean(axis=1), 'b-', linewidth=2, label='Avg Follower')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity')
        ax2.set_title('Velocity Tracking')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Tracking error
        ax3 = axes[1, 0]
        ax3.plot(time, result['errors'], 'g-', linewidth=1.5)
        ax3.axhline(y=result['avg_error'], color='r', linestyle='--', label=f'Avg: {result["avg_error"]:.4f}')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Tracking Error')
        ax3.set_title('Tracking Error over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Communication rate
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
        print(f"\nVisualization saved: {save_path}")
        
        return result


def main():
    """主函数"""
    if not Path(SAVE_MODEL_PATH).exists():
        print(f"Model file not found: {SAVE_MODEL_PATH}")
        return
    
    # Fixed trajectory evaluation
    print("\n" + "="*60)
    print("Fixed Trajectory Evaluation")
    print("   - Leader trajectory: Fixed (sine, A=2.0, omega=0.5)")
    print("   - Random seed: Fixed")
    print("="*60)
    
    evaluator_fixed = ModelEvaluator(
        use_fixed_seed=True, 
        enable_randomization=False
    )
    stats_fixed = evaluator_fixed.evaluate(num_episodes=5, deterministic=True, use_different_seeds=False)
    
    set_seed(SEED)
    evaluator_fixed.visualize(save_path='evaluation_fixed.png')
    
    # Random trajectory evaluation
    print("\n" + "="*60)
    print("Random Trajectory Evaluation (Generalization Test)")
    print("   - Leader trajectory: Randomized")
    print("   - Random seed: Different")
    print("="*60)
    
    evaluator_random = ModelEvaluator(
        use_fixed_seed=False,
        enable_randomization=True
    )
    stats_random = evaluator_random.evaluate(num_episodes=5, deterministic=True, use_different_seeds=True)
    
    evaluator_random.visualize(save_path='evaluation_random.png')
    
    # Summary
    print("\n" + "="*60)
    print("Evaluation Comparison Summary")
    print("="*60)
    print(f"  Fixed: R={stats_fixed['mean_reward']:.2f}+/-{stats_fixed['std_reward']:.2f}, "
          f"Err={stats_fixed['mean_error']:.4f}")
    print(f"  Random: R={stats_random['mean_reward']:.2f}+/-{stats_random['std_reward']:.2f}, "
          f"Err={stats_random['mean_error']:.4f}")
    print("="*60)


if __name__ == '__main__':
    main()
