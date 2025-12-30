"""有向生成树通信拓扑模块。

本模块实现了多智能体系统的有向生成树通信拓扑结构。
拓扑以领导者为根节点，通过有向边连接到跟随者智能体。

拓扑特点:
    - 领导者作为根节点，不接收任何信息
    - Pinned followers 直接与领导者相连，可获取领导者状态
    - 其他跟随者通过生成树结构间接获取领导者信息
    - 支持额外的随机边以增强连通性

Example:
    >>> from topology import DirectedSpanningTreeTopology
    >>> topology = DirectedSpanningTreeTopology(num_followers=6, num_pinned=2)
    >>> neighbors = topology.get_neighbors(node_id=1)
    >>> topology.visualize(save_path='topology.png')
"""
import torch
import numpy as np
from typing import List, Optional

from config import DEVICE, NUM_PINNED, TOPOLOGY_SEED


class DirectedSpanningTreeTopology:
    """有向生成树通信拓扑。

    构建以领导者为根的有向生成树，确保所有跟随者都能通过通信链路
    间接或直接获取领导者的状态信息。

    拓扑结构:
        Leader (0) -> Pinned Followers -> Other Followers

    Attributes:
        num_followers: 跟随者数量。
        num_agents: 总智能体数量（领导者 + 跟随者）。
        num_pinned: 直接连接领导者的跟随者数量。
        leader_id: 领导者节点 ID，固定为 0。
        pinned_followers: 直接连接领导者的跟随者 ID 列表。
        edge_index: 边索引张量，形状为 (2, num_edges)。
        in_degree: 各节点的入度。
        out_degree: 各节点的出度。
    """

    def __init__(self, num_followers: int, num_pinned: int = NUM_PINNED,
                 seed: int = TOPOLOGY_SEED) -> None:
        """初始化有向生成树拓扑。

        Args:
            num_followers: 跟随者智能体数量。
            num_pinned: 直接连接领导者的跟随者数量，
                不超过 num_followers。
            seed: 随机种子，用于生成可复现的拓扑结构。
        """
        self.num_followers = num_followers
        self.num_agents = num_followers + 1
        self.num_pinned = min(num_pinned, num_followers)
        self.leader_id = 0

        np.random.seed(seed)
        self._build_topology()

    def _build_topology(self) -> None:
        """构建有向生成树拓扑结构。

        算法步骤:
            1. 随机选择 pinned followers
            2. 建立领导者到 pinned followers 的边
            3. 为剩余跟随者随机选择已连接节点作为父节点
            4. 以一定概率添加额外边增强连通性
        """
        edges = []
        follower_ids = list(range(1, self.num_agents))

        # 随机选择 pinned followers
        self.pinned_followers = sorted(np.random.choice(
            follower_ids, self.num_pinned, replace=False
        ).tolist())

        # 领导者 -> pinned followers
        for f in self.pinned_followers:
            edges.append((self.leader_id, f))

        # 构建生成树：为未连接节点选择父节点
        unpinned = [f for f in follower_ids if f not in self.pinned_followers]
        connected = set(self.pinned_followers)

        for f in unpinned:
            parent = np.random.choice(list(connected))
            edges.append((parent, f))
            connected.add(f)

        # 添加额外边以增强连通性（概率 0.3）
        for f in follower_ids:
            potential_neighbors = [n for n in follower_ids if n != f]
            if potential_neighbors and np.random.random() < 0.3:
                neighbor = np.random.choice(potential_neighbors)
                if (neighbor, f) not in edges and (f, neighbor) not in edges:
                    edges.append((neighbor, f))

        src, dst = zip(*edges)
        self.edge_index = torch.tensor([src, dst], dtype=torch.long).to(DEVICE)
        self._compute_degrees()

    def _compute_degrees(self) -> None:
        """计算各节点的入度和出度。"""
        self.in_degree = torch.zeros(self.num_agents, device=DEVICE)
        self.out_degree = torch.zeros(self.num_agents, device=DEVICE)

        for i in range(self.edge_index.shape[1]):
            src, dst = self.edge_index[0, i].item(), self.edge_index[1, i].item()
            self.out_degree[src] += 1
            self.in_degree[dst] += 1

    def get_neighbors(self, node_id: int) -> List[int]:
        """获取指定节点的入边邻居列表。

        入边邻居是指向该节点发送信息的节点。

        Args:
            node_id: 目标节点 ID。

        Returns:
            入边邻居节点 ID 列表。
        """
        mask = self.edge_index[1] == node_id
        return self.edge_index[0, mask].tolist()

    def has_leader_access(self, node_id: int) -> bool:
        """检查节点是否直接连接领导者。

        Args:
            node_id: 目标节点 ID。

        Returns:
            如果节点是 pinned follower 则返回 True。
        """
        return node_id in self.pinned_followers

    def visualize(self, save_path: Optional[str] = None) -> None:
        """可视化拓扑结构。

        使用 networkx 和 matplotlib 绘制有向图，
        不同类型的节点使用不同颜色标识。

        Args:
            save_path: 图片保存路径。如果为 None，则仅显示不保存。

        Note:
            需要安装 matplotlib 和 networkx 库。
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
        except ImportError:
            print("Please install matplotlib and networkx")
            return

        G = nx.DiGraph()
        G.add_nodes_from(range(self.num_agents))

        edges = list(zip(
            self.edge_index[0].cpu().numpy(),
            self.edge_index[1].cpu().numpy()
        ))
        G.add_edges_from(edges)

        pos = nx.spring_layout(G, seed=42, k=2)
        pos[0] = np.array([0.5, 1.0])

        plt.figure(figsize=(10, 8))

        # 绘制不同类型的节点
        nx.draw_networkx_nodes(G, pos, nodelist=[0],
                               node_color='gold', node_size=800, label='Leader')
        nx.draw_networkx_nodes(G, pos, nodelist=self.pinned_followers,
                               node_color='lightgreen', node_size=500, label='Pinned')
        other_nodes = [n for n in range(1, self.num_agents)
                       if n not in self.pinned_followers]
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,
                               node_color='lightblue', node_size=400, label='Others')

        nx.draw_networkx_edges(G, pos, edge_color='gray',
                               arrows=True, arrowsize=20, alpha=0.7)

        labels = {0: 'L'}
        labels.update({i: f'F{i}' for i in range(1, self.num_agents)})
        nx.draw_networkx_labels(G, pos, labels, font_size=10)

        plt.title('Directed Spanning Tree Topology', fontsize=12)
        plt.legend(loc='upper left')
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

        print(f"\nTopology Statistics:")
        print(f"   Nodes: {self.num_agents}, Edges: {self.edge_index.shape[1]}")
        print(f"   Pinned Followers: {self.pinned_followers}")
