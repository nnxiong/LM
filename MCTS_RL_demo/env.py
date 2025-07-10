# env.py
import numpy as np
from typing import Tuple, List

class TicTacToe:
    def __init__(self):
        """
        初始化一个 3x3 的井字棋游戏。
        board: 9 个元素的一维数组，0 表示空位，1 表示玩家 X，-1 表示玩家 O
        player: 当前玩家（1 为 X，先手；-1 为 O）
        done: 游戏是否结束
        winner: 胜者（1=X，-1=O，0=平局）
        """
        self.board = np.zeros(9, dtype=int)  # 棋盘状态
        self.player = 1                      # 当前玩家（X 先手）
        self.done = False                    # 是否结束
        self.winner = None                   # 获胜者

    def reset(self) -> np.ndarray:
        """重置环境，返回初始状态"""
        self.board = np.zeros(9, dtype=int)
        self.player = 1
        self.done = False
        self.winner = None
        return self.board.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一个动作，并返回下一个状态、奖励、是否结束、其他信息

        参数：
            action (int): 动作位置（0~8）

        返回：
            next_state (np.ndarray): 下一状态
            reward (float): 奖励（+1 赢，-1 输，0 进行中）
            done (bool): 是否结束
            info (dict): 额外信息
        """
        if self.board[action] != 0 or self.done:
            return self.board.copy(), -10.0, True, {}  # 非法动作惩罚

        self.board[action] = self.player

        reward = 0.0
        if self.check_win():
            reward = 1.0
            self.done = True
            self.winner = self.player
        elif 0 not in self.board:
            self.done = True
            self.winner = 0  # 平局
        else:
            self.player *= -1  # 切换玩家

        return self.board.copy(), reward, self.done, {}

    def check_win(self) -> bool:
        """检查是否有玩家获胜"""
        win_pos = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],
            [0, 3, 6], [1, 4, 7], [2, 5, 8],
            [0, 4, 8], [2, 4, 6]
        ]
        for pos in win_pos:
            if self.board[pos[0]] == self.board[pos[1]] == self.board[pos[2]] != 0:
                return True
        return False

    def legal_moves(self) -> List[int]:
        """返回所有合法动作"""
        return np.where(self.board == 0)[0].tolist()