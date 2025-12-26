"""
配置文件 - 课程学习优化版（修复版）
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

# ==================== 课程学习参数（优化版）====================
COMM_PENALTY_INIT = 0.0
COMM_PENALTY_FINAL = 0.25
COMM_PENALTY_WARMUP = 150
COMM_PENALTY_ANNEAL = 200

THRESHOLD_MIN_INIT = 0.005
THRESHOLD_MAX_INIT = 0.05
THRESHOLD_MIN_FINAL = 0.05
THRESHOLD_MAX_FINAL = 0.40

COMM_BONUS_INIT = 0.1
COMM_BONUS_FINAL = 0.0

COMM_PENALTY = COMM_PENALTY_INIT

# ==================== 奖励参数 ====================
REWARD_MIN = -2.0
REWARD_MAX = 2.0
USE_SOFT_REWARD_SCALING = True

# ==================== SAC 参数 ====================
LEARNING_RATE = 3e-4
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
ALPHA_LR = 3e-4

GAMMA = 0.99
TAU = 0.005

INIT_ALPHA = 0.2
AUTO_ALPHA = True
TARGET_ENTROPY_RATIO = 0.5

HIDDEN_DIM = 256
BUFFER_SIZE = 500000
BATCH_SIZE = 512
GRADIENT_STEPS = 2

# ==================== 网络参数 ====================
LOG_STD_MIN = -20
LOG_STD_MAX = 2
U_SCALE = 5.0      # 🔧 控制输入缩放因子
TH_SCALE = 0.8     # 🔧 阈值输出缩放因子

# ==================== 训练参数 ====================
NUM_EPISODES = 500
VIS_INTERVAL = 10
NUM_PARALLEL_ENVS = 32
UPDATE_FREQUENCY = 4
USE_AMP = True

SAVE_MODEL_PATH = 'best_model.pt'


def get_comm_penalty(episode):
    """获取当前 episode 的通信惩罚系数"""
    if episode < COMM_PENALTY_WARMUP:
        return COMM_PENALTY_INIT
    elif episode < COMM_PENALTY_WARMUP + COMM_PENALTY_ANNEAL:
        progress = (episode - COMM_PENALTY_WARMUP) / COMM_PENALTY_ANNEAL
        return COMM_PENALTY_INIT + progress * (COMM_PENALTY_FINAL - COMM_PENALTY_INIT)
    else:
        return COMM_PENALTY_FINAL


def get_curriculum_progress(episode):
    """
    获取课程学习进度 [0, 1]
    """
    total_curriculum = COMM_PENALTY_WARMUP + COMM_PENALTY_ANNEAL
    if episode < COMM_PENALTY_WARMUP:
        return 0.0
    elif episode < total_curriculum:
        return (episode - COMM_PENALTY_WARMUP) / COMM_PENALTY_ANNEAL
    else:
        return 1.0


def get_threshold_bounds(episode):
    """
    获取当前 episode 的阈值边界
    """
    progress = get_curriculum_progress(episode)
    
    threshold_min = THRESHOLD_MIN_INIT + progress * (THRESHOLD_MIN_FINAL - THRESHOLD_MIN_INIT)
    threshold_max = THRESHOLD_MAX_INIT + progress * (THRESHOLD_MAX_FINAL - THRESHOLD_MAX_INIT)
    
    return threshold_min, threshold_max


def get_comm_bonus(episode):
    """
    获取当前 episode 的通信奖励系数
    """
    progress = get_curriculum_progress(episode)
    return COMM_BONUS_INIT * (1.0 - progress)


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
    print("🔧 Configuration - Curriculum Learning Optimized (Fixed)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Seed: {SEED}")
    print(f"  Agents: {NUM_AGENTS} (1 Leader + {NUM_FOLLOWERS} Followers)")
    print(f"  Pinned: {NUM_PINNED}")
    print(f"  Episodes: {NUM_EPISODES}, Max Steps: {MAX_STEPS}")
    print(f"  Parallel Envs: {NUM_PARALLEL_ENVS}")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  📚 Curriculum Learning:")
    print(f"     Comm Penalty: {COMM_PENALTY_INIT} → {COMM_PENALTY_FINAL}")
    print(f"     Threshold Range: [{THRESHOLD_MIN_INIT}, {THRESHOLD_MAX_INIT}] → [{THRESHOLD_MIN_FINAL}, {THRESHOLD_MAX_FINAL}]")
    print(f"     Comm Bonus: {COMM_BONUS_INIT} → {COMM_BONUS_FINAL}")
    print(f"     Warmup: {COMM_PENALTY_WARMUP} eps, Anneal: {COMM_PENALTY_ANNEAL} eps")
    print(f"  🔧 Action Scales: U={U_SCALE}, TH={TH_SCALE}")
    print("=" * 60)