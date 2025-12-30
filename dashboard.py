"""训练可视化仪表盘模块。

本模块提供 Jupyter Notebook 环境下的实时训练可视化功能。
"""
import time
import numpy as np
from typing import Dict, List, Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['figure.max_open_warning'] = 50
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    HAS_WIDGETS = True
except ImportError:
    HAS_WIDGETS = False

from config import MAX_STEPS


class TrainingDashboard:
    """训练仪表盘。

    Attributes:
        total_episodes: 总训练轮数。
        vis_interval: 可视化更新间隔。
        reward_history: 奖励历史记录。
        best_reward: 最佳奖励。
    """

    def __init__(self, total_episodes: int, vis_interval: int = 10,
                 max_steps: int = MAX_STEPS) -> None:
        """初始化仪表盘。

        Args:
            total_episodes: 总训练轮数。
            vis_interval: 可视化更新间隔。
            max_steps: 每轮最大步数。
        """
        self.total_episodes = total_episodes
        self.vis_interval = vis_interval
        self.max_steps = max_steps
        self.start_time: Optional[float] = None

        # 动态计算阈值
        self.reward_good_threshold = -0.33 * max_steps
        self.reward_poor_threshold = -1.0 * max_steps
        self.error_good_threshold = 0.2
        self.error_poor_threshold = 0.8
        self.comm_good_range = (0.2, 0.6)
        self.comm_poor_threshold = 0.8

        # 历史记录
        self.reward_history: List[float] = []
        self.tracking_error_history: List[float] = []
        self.comm_history: List[float] = []
        self.best_reward = -float('inf')
        self.best_trajectory: Optional[Dict] = None

        self.use_widgets = HAS_WIDGETS and HAS_MATPLOTLIB

        if self.use_widgets:
            self._create_widgets()
        else:
            print("Widgets not available, using console output")

    def _create_widgets(self) -> None:
        """创建 UI 组件。"""
        self.title_html = widgets.HTML(value=f"""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <h2 style="color: white; margin: 0; text-align: center;">
                    Leader-Follower MAS Control
                </h2>
            </div>
        """)

        self.main_progress = widgets.FloatProgress(
            value=0, min=0, max=100, description='Total:',
            bar_style='info', style={'bar_color': '#667eea', 'description_width': '60px'},
            layout=widgets.Layout(width='100%', height='25px')
        )

        self.step_progress = widgets.FloatProgress(
            value=0, min=0, max=100, description='Episode:',
            bar_style='success', style={'bar_color': '#764ba2', 'description_width': '60px'},
            layout=widgets.Layout(width='100%', height='20px')
        )

        self.progress_text = widgets.HTML(value="<p style='margin: 5px 0;'>Initializing...</p>")
        self.stats_html = widgets.HTML(value="")
        self.plot_output = widgets.Output(layout=widgets.Layout(min_height='400px'))
        self.log_output = widgets.Output(layout=widgets.Layout(
            height='150px', overflow='auto', border='1px solid #ddd', padding='10px', margin='5px 0'
        ))

    def _format_time(self, seconds: Optional[float]) -> str:
        """格式化时间显示。"""
        if seconds is None or seconds < 0:
            return "N/A"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m {int(seconds%60)}s"
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

    def _get_elapsed(self) -> float:
        """获取已用时间。"""
        return time.time() - self.start_time if self.start_time else 0

    def _estimate_remaining(self, episode: int) -> str:
        """估计剩余时间。"""
        elapsed = self._get_elapsed()
        if episode == 0 or elapsed <= 0:
            return "..."
        return self._format_time((elapsed / episode) * (self.total_episodes - episode))

    def _get_reward_color(self, reward: float) -> str:
        """获取奖励对应的颜色。"""
        if reward > self.reward_good_threshold:
            return "#48bb78"
        elif reward < self.reward_poor_threshold:
            return "#f56565"
        return "#ed8936"

    def display(self) -> None:
        """显示仪表盘。"""
        self.start_time = time.time()

        if self.use_widgets:
            dashboard_layout = widgets.VBox([
                self.title_html, self.main_progress, self.step_progress,
                self.progress_text, self.stats_html,
                widgets.HTML("<h4 style='margin: 15px 0 5px 0;'>Training Progress</h4>"),
                self.plot_output,
                widgets.HTML("<h4 style='margin: 15px 0 5px 0;'>Training Log</h4>"),
                self.log_output
            ])
            display(dashboard_layout)
        else:
            print("=" * 60)
            print("Training Started (Console Mode)")
            print("=" * 60)

    def update_step(self, step: int, max_steps: int) -> None:
        """更新步进度。"""
        if self.use_widgets:
            self.step_progress.value = (step / max_steps) * 100

    def update_episode(self, episode: int, reward: float, tracking_err: float,
                       comm: float, losses: Dict, trajectory_data: Optional[Dict] = None) -> None:
        """更新轮次信息。"""
        self.reward_history.append(reward)
        self.tracking_error_history.append(tracking_err)
        self.comm_history.append(comm)

        if reward > self.best_reward:
            self.best_reward = reward
            if trajectory_data is not None:
                self.best_trajectory = trajectory_data

        if self.use_widgets:
            self.main_progress.value = (episode / self.total_episodes) * 100
            self.step_progress.value = 0

            with self.log_output:
                ts = time.strftime("%H:%M:%S")
                status = "Best" if reward >= self.best_reward - 5 else (
                    "Good" if reward > self.reward_good_threshold * 1.5 else "Training")
                print(f"[{ts}] {status} Ep {episode:4d} | R:{reward:7.1f} | "
                      f"Err:{tracking_err:.4f} | Comm:{comm*100:.1f}%")

            if episode % self.vis_interval == 0 or episode == 1:
                self._update_plots()
        else:
            if episode % 20 == 0:
                print(f"Ep {episode:4d} | R:{reward:7.1f} | Err:{tracking_err:.4f} | Comm:{comm*100:.1f}%")

    def _update_plots(self) -> None:
        """更新图表。"""
        if not HAS_MATPLOTLIB or not self.use_widgets:
            return

        with self.plot_output:
            clear_output(wait=True)
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            num_eps = len(self.reward_history)

            # 位置跟踪
            ax1 = axes[0, 0]
            if self.best_trajectory is not None:
                t = self.best_trajectory['times']
                ax1.plot(t, self.best_trajectory['leader_pos'], 'r-', lw=2.5, label='Leader')
                ax1.plot(t, self.best_trajectory['follower_pos'].mean(axis=1), 'b--', lw=2, label='Avg Follower')
            ax1.set_title(f'Position (Best R={self.best_reward:.1f})', fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # 速度跟踪
            ax2 = axes[0, 1]
            if self.best_trajectory is not None:
                t = self.best_trajectory['times']
                ax2.plot(t, self.best_trajectory['leader_vel'], 'r-', lw=2.5, label='Leader')
                ax2.plot(t, self.best_trajectory['follower_vel'].mean(axis=1), 'b--', lw=2, label='Avg Follower')
            ax2.set_title('Velocity', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            # 奖励曲线
            ax3 = axes[1, 0]
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                ax3.plot(eps, self.reward_history, alpha=0.5, lw=1)
                if num_eps >= 10:
                    w = min(20, num_eps // 3)
                    if w >= 2:
                        sm = np.convolve(self.reward_history, np.ones(w)/w, mode='valid')
                        ax3.plot(np.arange(w, num_eps + 1), sm, lw=2.5, label='Smoothed')
            ax3.set_title('Reward', fontweight='bold')
            ax3.grid(True, alpha=0.3)

            # 跟踪误差
            ax4 = axes[1, 1]
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                ax4.plot(eps, self.tracking_error_history, alpha=0.5, lw=1)
            ax4.set_title('Tracking Error', fontweight='bold')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    def finish(self) -> None:
        """完成训练。"""
        elapsed = self._get_elapsed()

        if self.use_widgets:
            self.main_progress.value = 100
            self.main_progress.bar_style = 'success'
            with self.log_output:
                print("=" * 50)
                print(f"Training Complete! Time: {self._format_time(elapsed)}, Best: {self.best_reward:.2f}")
                print("=" * 50)
        else:
            print(f"\nTraining complete! Best reward: {self.best_reward:.2f}")

    def get_summary(self) -> Dict:
        """获取训练摘要。"""
        return {
            'best_reward': self.best_reward,
            'final_reward': self.reward_history[-1] if self.reward_history else None,
            'best_error': min(self.tracking_error_history) if self.tracking_error_history else None,
            'total_episodes': len(self.reward_history),
            'elapsed_time': self._get_elapsed()
        }
