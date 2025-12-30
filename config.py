"""多智能体系统共识控制配置模块。

本模块定义了 CTDE (Centralized Training Decentralized Execution) 架构下
领导者-跟随者多智能体系统的所有超参数和配置项。

配置分类:
    - 设备配置: GPU/CPU 设备选择和随机种子
    - 网络拓扑: 智能体数量和拓扑结构参数
    - 随机化配置: 领导者轨迹和跟随者状态的随机化范围
    - 环境参数: 状态/动作维度、时间步长、物理限制
    - 奖励参数: 跟踪惩罚、通信惩罚、改进奖励的权重
    - SAC 参数: 学习率、折扣因子、熵系数等
    - 训练参数: 训练轮数、并行环境数、更新频率

Example:
    >>> from config import DEVICE, NUM_AGENTS, set_seed
    >>> set_seed(42)
    >>> print(f"Using device: {DEVICE}, Agents: {NUM_AGENTS}")
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
MAX_STEPS = 300

LEADER_AMPLITUDE = 2.0
LEADER_OMEGA = 0.5
LEADER_PHASE = 0.0

POS_LIMIT = 10.0
VEL_LIMIT = 10.0

# ==================== 通信参数 ====================
THRESHOLD_MIN = 0.00
THRESHOLD_MAX = 1.0

# ==================== 奖励参数 ====================
REWARD_MIN = -2.0
REWARD_MAX = 2.0

# 跟踪惩罚参数
TRACKING_ERROR_SCALE = 1.5
TRACKING_PENALTY_MAX = 1.0

# 通信惩罚参数
COMM_PENALTY_BASE = 0.05
COMM_WEIGHT_DECAY = 1.0

# 改进奖励参数
IMPROVEMENT_SCALE = 0.5
IMPROVEMENT_CLIP = 0.1

# ==================== SAC 参数 ====================
LEARNING_RATE = 8e-5
ALPHA_LR = 3e-5
GAMMA = 0.99
TAU = 0.005
INIT_ALPHA = 0.6
AUTO_ALPHA = True
HIDDEN_DIM = 256
BUFFER_SIZE = 500000
BATCH_SIZE = 256
GRADIENT_STEPS = 1

# ==================== 网络参数 ====================
LOG_STD_MIN = -20
LOG_STD_MAX = 2
U_SCALE = 5.0
TH_SCALE = 1.0

# ==================== 训练参数 ====================
NUM_EPISODES = 1000
VIS_INTERVAL = 10
NUM_PARALLEL_ENVS = 16
UPDATE_FREQUENCY = 8
WARMUP_STEPS = 5000
USE_AMP = True
SAVE_MODEL_PATH = 'best_model.pt'

# ==================== CTDE 参数 ====================
USE_NEIGHBOR_INFO = True
NEIGHBOR_AGGREGATION = 'attention'
MAX_NEIGHBORS = 5
ACTOR_NUM_LAYERS = 3
CRITIC_NUM_LAYERS = 3


def set_seed(seed: int = SEED) -> None:
    """设置全局随机种子以确保实验可复现。

    同时设置 Python random、NumPy 和 PyTorch 的随机种子。
    如果 CUDA 可用，也会设置 CUDA 随机种子。

    Args:
        seed: 随机种子值，默认使用配置中的 SEED。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def print_config() -> None:
    """打印当前配置信息到控制台。

    以格式化的方式输出所有主要配置项，便于调试和记录实验设置。
    """
    print("=" * 60)
    print("Configuration - CTDE Architecture")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print()
    print("Multi-Agent System:")
    print(f"  Agents: {NUM_AGENTS} (1 Leader + {NUM_FOLLOWERS} Followers)")
    print(f"  Pinned Followers: {NUM_PINNED}")
    print(f"  State Dim: {STATE_DIM}, Action Dim: {ACTION_DIM}")
    print()
    print("Randomization:")
    print(f"  Enabled: {ENABLE_RANDOMIZATION}")
    if ENABLE_RANDOMIZATION:
        print(f"  Leader Amplitude Range: {LEADER_AMP_RANGE}")
        print(f"  Leader Omega Range: {LEADER_OMEGA_RANGE}")
        print(f"  Trajectory Types: {LEADER_TRAJECTORY_TYPES}")
    print()
    print("Environment:")
    print(f"  Max Steps: {MAX_STEPS}, DT: {DT}s")
    print(f"  Position Limit: +/-{POS_LIMIT}, Velocity Limit: +/-{VEL_LIMIT}")
    print()
    print("Reward Design:")
    print(f"  Tracking: -tanh(error * {TRACKING_ERROR_SCALE}) * {TRACKING_PENALTY_MAX}")
    print(f"  Comm Penalty Base: {COMM_PENALTY_BASE}")
    print(f"  Improvement Scale: {IMPROVEMENT_SCALE}, Clip: +/-{IMPROVEMENT_CLIP}")
    print()
    print("Communication:")
    print(f"  Threshold Range: [{THRESHOLD_MIN}, {THRESHOLD_MAX}]")
    print()
    print("CTDE Architecture:")
    print(f"  Actor Uses Neighbor Info: {USE_NEIGHBOR_INFO}")
    if USE_NEIGHBOR_INFO:
        print(f"  Neighbor Aggregation: {NEIGHBOR_AGGREGATION}")
    print()
    print("SAC Parameters:")
    print(f"  Learning Rate: {LEARNING_RATE}, Alpha LR: {ALPHA_LR}")
    print(f"  Init Alpha: {INIT_ALPHA}")
    print(f"  Gamma: {GAMMA}, Tau: {TAU}")
    print(f"  Hidden Dim: {HIDDEN_DIM}")
    print()
    print("Training:")
    print(f"  Episodes: {NUM_EPISODES}, Parallel Envs: {NUM_PARALLEL_ENVS}")
    print(f"  Batch Size: {BATCH_SIZE}, Buffer Size: {BUFFER_SIZE}")
    print(f"  Update Frequency: {UPDATE_FREQUENCY}, Warmup Steps: {WARMUP_STEPS}")
    print(f"  Use AMP: {USE_AMP}")
    print("=" * 60)


def get_config_dict() -> dict:
    """获取所有配置项的字典表示。

    Returns:
        包含所有配置项键值对的字典，可用于日志记录或序列化。
    """
    return {
        'device': str(DEVICE),
        'seed': SEED,
        'topology_seed': TOPOLOGY_SEED,
        'num_followers': NUM_FOLLOWERS,
        'num_pinned': NUM_PINNED,
        'num_agents': NUM_AGENTS,
        'enable_randomization': ENABLE_RANDOMIZATION,
        'leader_amp_range': LEADER_AMP_RANGE,
        'leader_omega_range': LEADER_OMEGA_RANGE,
        'leader_phase_range': LEADER_PHASE_RANGE,
        'leader_trajectory_types': LEADER_TRAJECTORY_TYPES,
        'follower_pos_init_std_range': FOLLOWER_POS_INIT_STD_RANGE,
        'follower_vel_init_std_range': FOLLOWER_VEL_INIT_STD_RANGE,
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'dt': DT,
        'max_steps': MAX_STEPS,
        'pos_limit': POS_LIMIT,
        'vel_limit': VEL_LIMIT,
        'reward_min': REWARD_MIN,
        'reward_max': REWARD_MAX,
        'tracking_error_scale': TRACKING_ERROR_SCALE,
        'tracking_penalty_max': TRACKING_PENALTY_MAX,
        'comm_penalty_base': COMM_PENALTY_BASE,
        'comm_weight_decay': COMM_WEIGHT_DECAY,
        'improvement_scale': IMPROVEMENT_SCALE,
        'improvement_clip': IMPROVEMENT_CLIP,
        'threshold_min': THRESHOLD_MIN,
        'threshold_max': THRESHOLD_MAX,
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
        'log_std_min': LOG_STD_MIN,
        'log_std_max': LOG_STD_MAX,
        'u_scale': U_SCALE,
        'th_scale': TH_SCALE,
        'num_episodes': NUM_EPISODES,
        'vis_interval': VIS_INTERVAL,
        'num_parallel_envs': NUM_PARALLEL_ENVS,
        'update_frequency': UPDATE_FREQUENCY,
        'warmup_steps': WARMUP_STEPS,
        'use_amp': USE_AMP,
        'use_neighbor_info': USE_NEIGHBOR_INFO,
        'neighbor_aggregation': NEIGHBOR_AGGREGATION,
        'max_neighbors': MAX_NEIGHBORS,
        'actor_num_layers': ACTOR_NUM_LAYERS,
        'critic_num_layers': CRITIC_NUM_LAYERS,
    }
