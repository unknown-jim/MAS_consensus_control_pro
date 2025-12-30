"""
训练可视化仪表盘
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
    """训练仪表盘"""
    
    def __init__(self, total_episodes: int, vis_interval: int = 10, 
                 max_steps: int = MAX_STEPS):
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
            print(f"Dynamic thresholds (MAX_STEPS={max_steps}):")
            print(f"   Reward: Good > {self.reward_good_threshold:.0f}, Poor < {self.reward_poor_threshold:.0f}")
    
    def _create_widgets(self) -> None:
        """创建 UI 组件"""
        self.title_html = widgets.HTML(value=f"""
            <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                        padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                <h2 style="color: white; margin: 0; text-align: center;">
                    Leader-Follower MAS Control
                </h2>
                <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0; text-align: center; font-size: 12px;">
                    MAX_STEPS={self.max_steps} | R_good>{self.reward_good_threshold:.0f} | R_poor<{self.reward_poor_threshold:.0f}
                </p>
            </div>
        """)
        
        self.main_progress = widgets.FloatProgress(
            value=0, min=0, max=100, description='Total:',
            bar_style='info', 
            style={'bar_color': '#667eea', 'description_width': '60px'},
            layout=widgets.Layout(width='100%', height='25px')
        )
        
        self.step_progress = widgets.FloatProgress(
            value=0, min=0, max=100, description='Episode:',
            bar_style='success', 
            style={'bar_color': '#764ba2', 'description_width': '60px'},
            layout=widgets.Layout(width='100%', height='20px')
        )
        
        self.progress_text = widgets.HTML(
            value="<p style='margin: 5px 0;'>Initializing...</p>"
        )
        
        self.stats_html = widgets.HTML(value="")
        
        self.plot_output = widgets.Output(
            layout=widgets.Layout(min_height='400px')
        )
        
        self.log_output = widgets.Output(
            layout=widgets.Layout(
                height='150px', 
                overflow='auto',
                border='1px solid #ddd', 
                padding='10px',
                margin='5px 0'
            )
        )
    
    def _format_time(self, seconds: Optional[float]) -> str:
        if seconds is None or seconds < 0:
            return "N/A"
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            return f"{int(seconds//60)}m {int(seconds%60)}s"
        return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"
    
    def _get_elapsed(self) -> float:
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def _estimate_remaining(self, episode: int) -> str:
        elapsed = self._get_elapsed()
        if episode == 0 or elapsed <= 0:
            return "..."
        return self._format_time((elapsed / episode) * (self.total_episodes - episode))
    
    def _get_reward_color(self, reward: float) -> str:
        if reward > self.reward_good_threshold:
            return "#48bb78"
        elif reward < self.reward_poor_threshold:
            return "#f56565"
        else:
            return "#ed8936"
    
    def _get_error_color(self, tracking_err: float) -> str:
        if tracking_err < self.error_good_threshold:
            return "#48bb78"
        elif tracking_err > self.error_poor_threshold:
            return "#f56565"
        else:
            return "#ed8936"
    
    def _get_comm_color(self, comm: float) -> str:
        if self.comm_good_range[0] <= comm <= self.comm_good_range[1]:
            return "#48bb78"
        elif comm > self.comm_poor_threshold:
            return "#f56565"
        else:
            return "#ed8936"
    
    def _generate_stats_html(self, episode: int, reward: float, tracking_err: float, 
                             comm: float, best: float, losses: Dict, elapsed: float) -> str:
        r_color = self._get_reward_color(reward)
        e_color = self._get_error_color(tracking_err)
        c_color = self._get_comm_color(comm)
        
        return f"""
        <div style="display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0;">
            <div style="flex:1;min-width:90px;background:linear-gradient(135deg,#667eea,#764ba2);padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:10px;opacity:0.9;">Episode</div>
                <div style="font-size:16px;font-weight:bold;">{episode}/{self.total_episodes}</div>
            </div>
            <div style="flex:1;min-width:90px;background:{r_color};padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:10px;opacity:0.9;">Reward</div>
                <div style="font-size:16px;font-weight:bold;">{reward:.1f}</div>
                <div style="font-size:9px;opacity:0.8;">Best: {best:.1f}</div>
            </div>
            <div style="flex:1;min-width:90px;background:{e_color};padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:10px;opacity:0.9;">Error</div>
                <div style="font-size:16px;font-weight:bold;">{tracking_err:.4f}</div>
            </div>
            <div style="flex:1;min-width:90px;background:{c_color};padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:10px;opacity:0.9;">Comm</div>
                <div style="font-size:16px;font-weight:bold;">{comm*100:.1f}%</div>
            </div>
            <div style="flex:1;min-width:90px;background:#4a5568;padding:10px;border-radius:8px;color:white;text-align:center;">
                <div style="font-size:10px;opacity:0.9;">Time</div>
                <div style="font-size:16px;font-weight:bold;">{self._format_time(elapsed)}</div>
                <div style="font-size:9px;opacity:0.8;">ETA: {self._estimate_remaining(episode)}</div>
            </div>
        </div>
        <div style="background:#f0f4f8;padding:8px;border-radius:6px;font-size:11px;margin-top:5px;">
            <span style="color:#666;">Q Loss:</span> <b>{losses.get('q1', 0):.4f}</b> |
            <span style="color:#666;">Actor Loss:</span> <b>{losses.get('actor', 0):.4f}</b> |
            <span style="color:#666;">alpha:</span> <b>{losses.get('alpha', 0.2):.3f}</b>
        </div>
        """
    
    def display(self) -> None:
        self.start_time = time.time()
        
        if self.use_widgets:
            dashboard_layout = widgets.VBox([
                self.title_html,
                self.main_progress,
                self.step_progress,
                self.progress_text,
                self.stats_html,
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
        if self.use_widgets:
            self.step_progress.value = (step / max_steps) * 100
    
    def update_episode(self, episode: int, reward: float, tracking_err: float, 
                       comm: float, losses: Dict, trajectory_data: Optional[Dict] = None) -> None:
        elapsed = self._get_elapsed()
        
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
            
            speed = episode / elapsed if elapsed > 0 else 0
            self.progress_text.value = f"""
            <p style='margin: 5px 0; font-size: 13px;'>
                <b>Episode {episode}/{self.total_episodes}</b> | 
                Speed: <b>{speed:.2f}</b> ep/s
            </p>
            """
            
            self.stats_html.value = self._generate_stats_html(
                episode, reward, tracking_err, comm, 
                self.best_reward, losses, elapsed
            )
            
            with self.log_output:
                ts = time.strftime("%H:%M:%S")
                if reward >= self.best_reward - 5:
                    status = "Best"
                elif reward > self.reward_good_threshold * 1.5:
                    status = "Good"
                else:
                    status = "Training"
                print(f"[{ts}] {status} Ep {episode:4d} | R:{reward:7.1f} | "
                      f"Err:{tracking_err:.4f} | Comm:{comm*100:.1f}%")
            
            if episode % self.vis_interval == 0 or episode == 1:
                self._update_plots()
        else:
            if episode % 20 == 0:
                print(f"Ep {episode:4d} | R:{reward:7.1f} | Err:{tracking_err:.4f} | Comm:{comm*100:.1f}%")
    
    def _update_plots(self) -> None:
        if not HAS_MATPLOTLIB or not self.use_widgets:
            return
        
        with self.plot_output:
            clear_output(wait=True)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 9))
            
            leader_color = '#e74c3c'
            smooth_color = '#667eea'
            raw_color = '#bdc3c7'
            
            num_eps = len(self.reward_history)
            
            # Position tracking
            ax1 = axes[0, 0]
            if self.best_trajectory is not None:
                t = self.best_trajectory['times']
                fp = self.best_trajectory['follower_pos']
                lp = self.best_trajectory['leader_pos']
                
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, fp.shape[1]))
                for i in range(fp.shape[1]):
                    ax1.plot(t, fp[:, i], color=colors[i], alpha=0.6, lw=1.0)
                
                ax1.plot(t, lp, color=leader_color, lw=2.5, label='Leader')
                ax1.plot(t, fp.mean(axis=1), color='#2ecc71', lw=2, linestyle='--', 
                        label='Avg Follower', alpha=0.8)
            
            ax1.set_title(f'Position (Best R={self.best_reward:.1f})', fontweight='bold')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Velocity tracking
            ax2 = axes[0, 1]
            if self.best_trajectory is not None:
                t = self.best_trajectory['times']
                fv = self.best_trajectory['follower_vel']
                lv = self.best_trajectory['leader_vel']
                
                colors = plt.cm.Blues(np.linspace(0.4, 0.9, fv.shape[1]))
                for i in range(fv.shape[1]):
                    ax2.plot(t, fv[:, i], color=colors[i], alpha=0.6, lw=1.0)
                
                ax2.plot(t, lv, color=leader_color, lw=2.5, label='Leader')
                ax2.plot(t, fv.mean(axis=1), color='#2ecc71', lw=2, linestyle='--', 
                        label='Avg Follower', alpha=0.8)
            
            ax2.set_title('Velocity', fontweight='bold')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # Reward curve
            ax3 = axes[1, 0]
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                ax3.plot(eps, self.reward_history, color=raw_color, alpha=0.5, lw=1)
                
                if num_eps >= 10:
                    w = min(20, num_eps // 3)
                    if w >= 2:
                        sm = np.convolve(self.reward_history, np.ones(w)/w, mode='valid')
                        ax3.plot(np.arange(w, num_eps + 1), sm, color=smooth_color, lw=2.5, 
                                label='Smoothed')
                
                best_idx = np.argmax(self.reward_history)
                ax3.scatter([best_idx + 1], [self.reward_history[best_idx]], 
                           color='gold', s=150, marker='*', zorder=15,
                           edgecolors='black', linewidths=0.5)
                
                ax3.set_xlim(0, max(num_eps + 1, 10))
                
                ax3.axhline(y=self.reward_good_threshold, color='green', linestyle='--', 
                           alpha=0.5, label=f'Good (>{self.reward_good_threshold:.0f})')
                ax3.axhline(y=self.reward_poor_threshold, color='red', linestyle='--', 
                           alpha=0.5, label=f'Poor (<{self.reward_poor_threshold:.0f})')
            
            ax3.set_title('Reward', fontweight='bold')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Reward')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            if num_eps > 0:
                ax3t = ax3.twinx()
                eps = np.arange(1, num_eps + 1)
                ax3t.plot(eps, [c*100 for c in self.comm_history], 
                         color='#e74c3c', linestyle=':', lw=1.5, alpha=0.7)
                ax3t.set_ylabel('Comm Rate (%)', color='#e74c3c')
                ax3t.set_ylim(0, 100)
                ax3t.tick_params(axis='y', labelcolor='#e74c3c')
            
            # Tracking error
            ax4 = axes[1, 1]
            if num_eps > 0:
                eps = np.arange(1, num_eps + 1)
                ax4.plot(eps, self.tracking_error_history, color='#f39c12', alpha=0.5, lw=1)
                
                if num_eps >= 10:
                    w = min(20, num_eps // 3)
                    if w >= 2:
                        sme = np.convolve(self.tracking_error_history, np.ones(w)/w, mode='valid')
                        ax4.plot(np.arange(w, num_eps + 1), sme, color='#27ae60', lw=2.5,
                                label='Smoothed')
                
                min_idx = np.argmin(self.tracking_error_history)
                ax4.scatter([min_idx + 1], [self.tracking_error_history[min_idx]], 
                           color='lime', s=150, marker='*', zorder=15,
                           edgecolors='black', linewidths=0.5)
                
                ax4.set_xlim(0, max(num_eps + 1, 10))
                
                ax4.axhline(y=self.error_good_threshold, color='green', linestyle='--', 
                           alpha=0.5, label=f'Excellent (<{self.error_good_threshold})')
                ax4.axhline(y=self.error_poor_threshold, color='red', linestyle='--', 
                           alpha=0.5, label=f'Poor (>{self.error_poor_threshold})')
            
            ax4.set_title('Tracking Error', fontweight='bold')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Error')
            ax4.legend(loc='upper right', fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
    
    def finish(self) -> None:
        elapsed = self._get_elapsed()
        
        if self.use_widgets:
            self.main_progress.value = 100
            self.main_progress.bar_style = 'success'
            
            with self.log_output:
                print("=" * 50)
                print(f"Training Complete!")
                print(f"   Total Time: {self._format_time(elapsed)}")
                print(f"   Best Reward: {self.best_reward:.2f}")
                if self.tracking_error_history:
                    print(f"   Best Error: {min(self.tracking_error_history):.4f}")
                if self.comm_history:
                    print(f"   Final Comm Rate: {self.comm_history[-1]*100:.1f}%")
                print("=" * 50)
        else:
            print(f"\nTraining complete! Best reward: {self.best_reward:.2f}")
    
    def get_summary(self) -> Dict:
        return {
            'best_reward': self.best_reward,
            'final_reward': self.reward_history[-1] if self.reward_history else None,
            'best_error': min(self.tracking_error_history) if self.tracking_error_history else None,
            'final_comm_rate': self.comm_history[-1] if self.comm_history else None,
            'total_episodes': len(self.reward_history),
            'elapsed_time': self._get_elapsed()
        }
