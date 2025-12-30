# 🎯 多智能体系统事件触发共识控制

基于深度强化学习的领导者-跟随者多智能体系统（MAS）事件触发共识控制项目。

## 📋 项目简介

本项目实现了一个基于 **CTDE（集中训练分布式执行）** 架构的多智能体共识控制系统，采用 **SAC（Soft Actor-Critic）** 算法训练智能体学习最优的控制策略和事件触发通信机制。

### 核心特性

- **领导者-跟随者架构**：1 个领导者 + N 个跟随者的分层控制结构
- **有向生成树拓扑**：保证通信网络的连通性和高效性
- **事件触发通信**：智能体自主决定何时通信，降低通信负载
- **域随机化训练**：支持多种领导者轨迹类型，提升策略泛化能力
- **并行环境训练**：多环境并行采样，加速训练过程

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CTDE 架构                                 │
├─────────────────────────────────────────────────────────────┤
│  训练阶段（集中式）          执行阶段（分布式）              │
│  ┌─────────────────┐        ┌─────────────────┐            │
│  │ Centralized     │        │ Decentralized   │            │
│  │ Critic          │        │ Actor           │            │
│  │ (全局状态)       │        │ (局部观测+邻居) │            │
│  └─────────────────┘        └─────────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 智能体动作空间

每个跟随者智能体输出两个动作：
- **控制输入 u**：连续控制信号，范围 [-5, 5]
- **通信阈值 θ**：事件触发阈值，范围 [0, 1]

## 📁 项目结构

```
MAS_consensus_control_pro/
├── config.py          # 配置模块 - 所有超参数
├── topology.py        # 拓扑模块 - 有向生成树通信拓扑
├── networks.py        # 网络模块 - Actor/Critic 神经网络
├── buffer.py          # 缓冲区模块 - 经验回放
├── environment.py     # 环境模块 - 领导者-跟随者 MAS 环境
├── agent.py           # 智能体模块 - SAC 智能体
├── utils.py           # 工具模块 - 辅助函数
├── train.py           # 训练模块 - 训练循环
├── evaluate.py        # 评估模块 - 策略评估
├── dashboard.py       # 可视化模块 - 训练仪表盘
├── main.ipynb         # 主程序 - Jupyter Notebook 入口
└── best_model.pt      # 最优模型权重
```

## 🚀 快速开始

### 环境要求

- Python >= 3.8
- PyTorch >= 1.12
- NumPy
- Matplotlib
- NetworkX
- tqdm

### 安装依赖

```bash
pip install torch numpy matplotlib networkx tqdm
```

### 运行训练

**方式一：使用 Jupyter Notebook**

```bash
jupyter notebook main.ipynb
```

**方式二：使用 Python 脚本**

```python
from train import train
from config import NUM_EPISODES, VIS_INTERVAL

# 开始训练
agent, topology, dashboard = train(
    num_episodes=NUM_EPISODES,
    vis_interval=VIS_INTERVAL
)
```

### 评估模型

```python
from utils import plot_evaluation
from topology import DirectedSpanningTreeTopology
from agent import SACAgent

# 加载模型
topology = DirectedSpanningTreeTopology(6, num_pinned=2)
agent = SACAgent(topology)
agent.load('best_model.pt')

# 可视化评估
plot_evaluation(agent, topology, num_tests=3, save_path='evaluation.png')
```

## ⚙️ 配置说明

主要配置项位于 `config.py`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `NUM_FOLLOWERS` | 6 | 跟随者数量 |
| `NUM_PINNED` | 2 | 能直接获取领导者信息的跟随者数量 |
| `MAX_STEPS` | 300 | 每回合最大步数 |
| `NUM_EPISODES` | 1000 | 训练回合数 |
| `NUM_PARALLEL_ENVS` | 16 | 并行环境数量 |
| `LEARNING_RATE` | 8e-5 | Actor/Critic 学习率 |
| `HIDDEN_DIM` | 256 | 隐藏层维度 |
| `BATCH_SIZE` | 256 | 批次大小 |
| `BUFFER_SIZE` | 500000 | 经验回放缓冲区大小 |

## 📊 奖励设计

奖励函数由三部分组成：

1. **跟踪惩罚**：惩罚跟随者与领导者之间的状态误差
   ```
   r_tracking = -tanh(error * scale) * max_penalty
   ```

2. **通信惩罚**：惩罚过多的通信行为
   ```
   r_comm = -base_penalty * comm_rate
   ```

3. **改进奖励**：奖励误差的减小
   ```
   r_improve = clip(prev_error - curr_error, -clip, +clip) * scale
   ```

## 🔬 算法细节

### SAC 算法

采用 Soft Actor-Critic 算法，具有以下特点：
- 最大熵强化学习框架
- 自动熵系数调节
- 双 Q 网络减少过估计
- 目标网络软更新

### 事件触发机制

智能体输出的阈值 θ 用于事件触发判断：
- 当 `||x_i - x_hat_i|| > θ` 时触发通信
- θ 越大，通信越稀疏
- 智能体需学习在跟踪精度和通信效率间权衡

## 📈 训练监控

训练过程中会实时显示：
- 回合奖励曲线
- 跟踪误差曲线
- 通信率曲线
- 智能体轨迹可视化

## 📝 引用

如果本项目对您的研究有帮助，欢迎引用。

## 📄 许可证

MIT License
