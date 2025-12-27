"""
配置文件 - CTDE 架构版（修复策略崩溃）
"""
import torch
import random
import numpy as np

# ==================== 设备配置 ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42
TOPOLOGY_SEED = 42

# ==================== 网络拓扑 ====================
NUM_FOLLOWERS = 6
NUM_PINNED = 2
NUM_AGENTS = NUM_FOLLOWERS + 1

# ==================== 随机化配置 ====================
ENABLE_RANDOMIZATION = True

# 领导者动力学随机化范围
LEADER_AMP_RANGE = (1.0, 3.0)
LEADER_OMEGA_RANGE = (0.3, 0.8)
LEADER_PHASE_RANGE = (0.0, 2 * np.pi)
LEADER_TRAJECTORY_TYPES = ['sine', 'cosine', 'mixed', 'chirp']

# 跟随者状态随机化范围
FOLLOWER_POS_INIT_STD_RANGE = (0.1, 0.5)
FOLLOWER_VEL_INIT_STD_RANGE = (0.05, 0.2)

# ==================== 环境参数 ====================
STATE_DIM = 4
ACTION_DIM = 2
DT = 0.05
MAX_STEPS = 300  # ✅ 减少步数，加快迭代

LEADER_AMPLITUDE = 2.0
LEADER_OMEGA = 0.5
LEADER_PHASE = 0.0

POS_LIMIT = 10.0
VEL_LIMIT = 10.0

# ==================== 通信参数 ====================
THRESHOLD_MIN = 0.00
THRESHOLD_MAX = 1.0

# ==================== 奖励参数（修复版）====================
REWARD_MIN = -2.0
REWARD_MAX = 2.0
USE_SOFT_REWARD_SCALING = False  # ✅ 关闭软缩放，使用硬裁剪

# ---------- 跟踪惩罚参数 ----------
TRACKING_ERROR_SCALE = 1.5  # ✅ 降低敏感度
TRACKING_PENALTY_MAX = 1.0  # ✅ 降低最大惩罚

# ---------- 通信惩罚参数 ----------
COMM_PENALTY_BASE = 0.05  # ✅ 增加通信成本
COMM_WEIGHT_DECAY = 1.0   # ✅ 减缓通信优化压力（原2.0）

# ---------- 改进奖励参数 ----------
IMPROVEMENT_SCALE = 0.5   # ✅ 大幅降低改进奖励权重
IMPROVEMENT_CLIP = 0.1    # ✅ 更严格的裁剪

# ==================== SAC 参数（方案三：稳定基础上提升性能）====================
LEARNING_RATE = 8e-5      # ✅ 略微提高学习率（原5e-5）
ALPHA_LR = 3e-5           # ✅ 保持低 alpha 学习率
GAMMA = 0.99
TAU = 0.005
INIT_ALPHA = 0.6          # ✅ 减少探索加速收敛（原0.8）
AUTO_ALPHA = True
TARGET_ENTROPY_SCALE = 1.0  # ✅ 新增：目标熵缩放因子
HIDDEN_DIM = 256
BUFFER_SIZE = 500000
BATCH_SIZE = 256          # ✅ 减小批次大小
GRADIENT_STEPS = 1        # ✅ 减少梯度步数

# ==================== 网络参数 ====================
LOG_STD_MIN = -20
LOG_STD_MAX = 2
U_SCALE = 5.0
TH_SCALE = 1.0

# ==================== 训练参数（方案三：延长训练）====================
NUM_EPISODES = 600        # ✅ 延长训练观察收敛（原450）
VIS_INTERVAL = 10
NUM_PARALLEL_ENVS = 16    # ✅ 减少并行环境数
UPDATE_FREQUENCY = 8      # ✅ 降低更新频率
WARMUP_STEPS = 5000       # ✅ 新增：预热步数
USE_AMP = True
SAVE_MODEL_PATH = 'best_model.pt'

# ==================== CTDE 参数 ====================
USE_CTDE = True
USE_NEIGHBOR_INFO = True
NEIGHBOR_AGGREGATION = 'attention'
MAX_NEIGHBORS = 5
ACTOR_NUM_LAYERS = 3
CRITIC_NUM_LAYERS = 3
USE_ATTENTION_CRITIC = False


def set_seed(seed=SEED):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_config():
    """打印配置信息"""
    print("=" * 60)
    print("🔧 Configuration - CTDE with Adaptive Reward (Fixed)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print()
    print("📊 Multi-Agent System:")
    print(f"  Agents: {NUM_AGENTS} (1 Leader + {NUM_FOLLOWERS} Followers)")
    print(f"  Pinned Followers: {NUM_PINNED}")
    print(f"  State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")
    print()
    print("🎲 Randomization:")
    print(f"  Enabled: {ENABLE_RANDOMIZATION}")
    if ENABLE_RANDOMIZATION:
        print(f"  Leader Amplitude Range: {LEADER_AMP_RANGE}")
        print(f"  Leader Omega Range: {LEADER_OMEGA_RANGE}")
        print(f"  Trajectory Types: {LEADER_TRAJECTORY_TYPES}")
    print()
    print("🎮 Environment:")
    print(f"  Max Steps: {MAX_STEPS}, DT: {DT}s")
    print(f"  Position Limit: ±{POS_LIMIT}, Velocity Limit: ±{VEL_LIMIT}")
    print()
    print("🎁 Reward Design (Fixed):")
    print(f"  Soft Scaling: {USE_SOFT_REWARD_SCALING}")
    print(f"  Tracking: -tanh(error * {TRACKING_ERROR_SCALE}) * {TRACKING_PENALTY_MAX}")
    print(f"  Comm Penalty Base: {COMM_PENALTY_BASE}")
    print(f"  Improvement Scale: {IMPROVEMENT_SCALE}, Clip: ±{IMPROVEMENT_CLIP}")
    print()
    print("📡 Communication:")
    print(f"  Threshold Range: [{THRESHOLD_MIN}, {THRESHOLD_MAX}]")
    print()
    print("🏗️ CTDE Architecture:")
    print(f"  Actor Uses Neighbor Info: {USE_NEIGHBOR_INFO}")
    if USE_NEIGHBOR_INFO:
        print(f"  Neighbor Aggregation: {NEIGHBOR_AGGREGATION}")
    print()
    print("🧠 SAC Parameters (Fixed):")
    print(f"  Learning Rate: {LEARNING_RATE}, Alpha LR: {ALPHA_LR}")
    print(f"  Init Alpha: {INIT_ALPHA}")
    print(f"  Gamma: {GAMMA}, Tau: {TAU}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print()
    print("🚀 Training (Fixed):")
    print(f"  Episodes: {NUM_EPISODES}, Parallel Envs: {NUM_PARALLEL_ENVS}")
    print(f"  Batch Size: {BATCH_SIZE}, Buffer Size: {BUFFER_SIZE}")
    print(f"  Update Frequency: {UPDATE_FREQUENCY}, Warmup Steps: {WARMUP_STEPS}")
    print(f"  Use AMP: {USE_AMP}")
    print("=" * 60)


def get_config_dict():
    """获取配置字典"""
    return {
        # 设备和种子
        'device': str(DEVICE),
        'seed': SEED,
        'topology_seed': TOPOLOGY_SEED,
        
        # 网络拓扑
        'num_followers': NUM_FOLLOWERS,
        'num_pinned': NUM_PINNED,
        'num_agents': NUM_AGENTS,
        
        # 随机化配置
        'enable_randomization': ENABLE_RANDOMIZATION,
        'leader_amp_range': LEADER_AMP_RANGE,
        'leader_omega_range': LEADER_OMEGA_RANGE,
        'leader_phase_range': LEADER_PHASE_RANGE,
        'leader_trajectory_types': LEADER_TRAJECTORY_TYPES,
        'follower_pos_init_std_range': FOLLOWER_POS_INIT_STD_RANGE,
        'follower_vel_init_std_range': FOLLOWER_VEL_INIT_STD_RANGE,
        
        # 环境参数
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'dt': DT,
        'max_steps': MAX_STEPS,
        'pos_limit': POS_LIMIT,
        'vel_limit': VEL_LIMIT,
        
        # 奖励参数
        'reward_min': REWARD_MIN,
        'reward_max': REWARD_MAX,
        'use_soft_reward_scaling': USE_SOFT_REWARD_SCALING,
        'tracking_error_scale': TRACKING_ERROR_SCALE,
        'tracking_penalty_max': TRACKING_PENALTY_MAX,
        'comm_penalty_base': COMM_PENALTY_BASE,
        'comm_weight_decay': COMM_WEIGHT_DECAY,
        'improvement_scale': IMPROVEMENT_SCALE,
        'improvement_clip': IMPROVEMENT_CLIP,
        
        # 通信参数
        'threshold_min': THRESHOLD_MIN,
        'threshold_max': THRESHOLD_MAX,
        
        # SAC 参数
        'learning_rate': LEARNING_RATE,
        'alpha_lr': ALPHA_LR,
        'gamma': GAMMA,
        'tau': TAU,
        'init_alpha': INIT_ALPHA,
        'auto_alpha': AUTO_ALPHA,
        'hidden_dim': HIDDEN_DIM,
        'buffer_size': BUFFER_SIZE,
        'batch_size': BATCH_SIZE,
        'gradient_steps': GRADIENT_STEPS,
        
        # 网络参数
        'log_std_min': LOG_STD_MIN,
        'log_std_max': LOG_STD_MAX,
        'u_scale': U_SCALE,
        'th_scale': TH_SCALE,
        
        # 训练参数
        'num_episodes': NUM_EPISODES,
        'vis_interval': VIS_INTERVAL,
        'num_parallel_envs': NUM_PARALLEL_ENVS,
        'update_frequency': UPDATE_FREQUENCY,
        'warmup_steps': WARMUP_STEPS,
        'use_amp': USE_AMP,
        
        # CTDE 参数
        'use_ctde': USE_CTDE,
        'use_neighbor_info': USE_NEIGHBOR_INFO,
        'neighbor_aggregation': NEIGHBOR_AGGREGATION,
        'max_neighbors': MAX_NEIGHBORS,
        'actor_num_layers': ACTOR_NUM_LAYERS,
        'critic_num_layers': CRITIC_NUM_LAYERS,
    }