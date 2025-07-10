# mcts.py
import math
import random
from typing import List, Tuple, Optional
import torch
import numpy as np

from env import TicTacToe
from model import PolicyValueNet

class Node:
    def __init__(self, state: np.ndarray, parent: Optional['Node'] = None, prior: float = 0.0):
        """
        树节点类，表示一个状态及其统计信息

        参数：
            state (np.ndarray): 当前棋盘状态
            parent (Node): 父节点
            prior (float): 来自策略网络的先验概率
        """
        self.state = state
        self.parent = parent
        self.children = {}
        self.prior = prior
        self.visit_count = 0
        self.total_value = 0.0
        self.mean_value = 0.0

    def expand(self, net: PolicyValueNet):
        """
        使用策略网络扩展子节点

        参数：
            net (PolicyValueNet): 策略价值网络
        """
        game = TicTacToe()
        game.board = self.state.copy()
        legal_actions = game.legal_moves()
        state_tensor = torch.tensor(self.state, dtype=torch.float32)
        with torch.no_grad():
            logits, _ = net(state_tensor)
        probs = torch.softmax(logits, dim=-1).numpy()

        for action in legal_actions:
            next_state = self.state.copy()
            next_state[action] = self._current_player()
            self.children[action] = Node(next_state, parent=self, prior=probs[action])

    def evaluate(self, net: PolicyValueNet) -> float:
        """
        使用价值网络评估当前状态

        参数：
            net (PolicyValueNet): 策略价值网络

        返回：
            value (float): 当前状态价值 [-1, 1]
        """
        state_tensor = torch.tensor(self.state, dtype=torch.float32)
        with torch.no_grad():
            _, value = net(state_tensor)
        return value.item()

    def is_leaf(self) -> bool:
        """是否是叶子节点"""
        return len(self.children) == 0

    def is_terminal(self) -> bool:
        """是否是终止状态"""
        game = TicTacToe()
        game.board = self.state.copy()
        return game.check_win() or 0 not in self.state

    def _current_player(self) -> int:
        """获取当前玩家（用于模拟）"""
        return 1 if sum(self.state == 1) == sum(self.state == -1) else -1


def ucb_score(node: Node, c: float = 1.0) -> float:
    """
    UCB 公式计算节点得分，平衡探索与利用

    参数：
        node (Node): 当前节点
        c (float): 探索系数

    返回：
        score (float): UCB 得分
    """
    pb_c = math.log((node.parent.visit_count + 1e-6) / (node.visit_count + 1e-6)) * c
    prior_score = pb_c * node.prior
    value_score = -node.mean_value  # 反转对手视角
    return prior_score + value_score


def select_leaf(root: Node, net: PolicyValueNet) -> Node:
    """
    选择最有潜力的叶节点进行扩展

    参数：
        root (Node): 根节点
        net (PolicyValueNet): 策略价值网络

    返回：
        leaf (Node): 叶节点
    """
    node = root
    while not node.is_leaf():
        best_score = -float('inf')
        best_child = None
        for a, child in node.children.items():
            score = ucb_score(child, c=1.0)
            if score > best_score:
                best_score = score
                best_child = child
        if best_child is None:
            break
        node = best_child
    if node.is_leaf() and not node.is_terminal():
        node.expand(net)
    return node


def backpropagate(node: Node, value: float):
    """
    回溯更新路径上的所有节点统计信息

    参数：
        node (Node): 叶节点
        value (float): 最终评估值
    """
    while node:
        node.visit_count += 1
        node.total_value += value
        node.mean_value = node.total_value / node.visit_count
        value = -value  # 对方视角
        node = node.parent


# def get_action_probs(root: Node, temp: float = 1e-3) -> List[float]:
#     """
#     根据访问次数生成动作概率分布

#     参数：
#         root (Node): 根节点
#         temp (float): 温度系数，控制探索强度

#     返回：
#         probs (List[float]): 各动作的概率
#     """
#     counts = [root.children[a].visit_count if a in root.children else 0 for a in range(9)]
#     counts = [c ** (1/temp) for c in counts]
#     total = sum(counts)
#     return [c / total for c in counts]


def get_action_probs(root: Node, temp=1.0):
    logits = np.array([child.visit_count for child in root.children.values()])
    logits = logits / (temp + 1e-8)  # 防止除零
    probs = np.exp(logits - np.max(logits))  # 数值稳定
    probs /= probs.sum()
    full_probs = [0] * 9
    for idx, action in enumerate(root.children.keys()):
        full_probs[action] = probs[idx]
    return full_probs

# def get_action_probs_with_noise(root: Node, temp=1e-3, noise_eps=0.25):
# def get_action_probs(root: Node, temp=1e-3, noise_eps=0.25):
#     counts = [root.children[a].visit_count if a in root.children else 0 for a in range(9)]
#     legal_actions = [a for a in range(9) if counts[a] > 0]

#     if not legal_actions:
#         return [1/9]*9

#     counts = np.array(counts, dtype=np.float32)
#     counts = counts / counts.sum()
#     noise = np.random.dirichlet([0.3] * len(legal_actions))
#     counts = [(1 - noise_eps) * c + noise_eps * n for c, n in zip(counts, noise)]

#     total = sum(counts)
#     return [c / total for c in counts]