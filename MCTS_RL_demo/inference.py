# inference.py

from typing import Tuple
import numpy as np
import torch
from env import TicTacToe
from model import PolicyValueNet
from mcts import Node, select_leaf, backpropagate, get_action_probs

def load_model(model_path: str) -> PolicyValueNet:
    """
    加载训练好的模型权重

    参数：
        model_path (str): 模型保存路径

    返回：
        net (PolicyValueNet): 加载后的模型
    """
    net = PolicyValueNet()
    net.load_state_dict(torch.load(model_path))
    net.eval()  # 设置为评估模式
    return net

def human_move(state: np.ndarray) -> int:
    """
    获取人类玩家的输入动作

    参数：
        state (np.ndarray): 当前棋盘状态

    返回：
        action (int): 玩家选择的位置
    """
    print("当前棋盘：")
    board = state.reshape(3, 3)
    for row in board:
        print(" | ".join(["X" if x == 1 else "O" if x == -1 else " " for x in row]))
    while True:
        try:
            action = int(input("请输入你要落子的位置 (0-8): "))
            if action in range(9) and state[action] == 0:
                return action
            else:
                print("无效位置，请重新输入！")
        except ValueError:
            print("请输入数字！")

def ai_move(state: np.ndarray, net: PolicyValueNet) -> Tuple[int, float]:
    """
    使用 MCTS 让 AI 选择动作

    参数：
        state (np.ndarray): 当前棋盘状态
        net (PolicyValueNet): 训练好的模型

    返回：
        action (int): AI 选择的动作
        win_rate (float): AI 胜率估计
    """
    root = Node(state)
    root.expand(net)

    for _ in range(50):  # MCTS 搜索次数
        leaf = select_leaf(root, net)
        value = leaf.evaluate(net)
        backpropagate(leaf, value)

    probs = get_action_probs(root)
    action = np.argmax(probs)
    win_rate = root.children[action].mean_value if action in root.children else 0.0
    return action, win_rate

def play_against_ai(model_path: str):
    """
    与训练好的 AI 对弈

    参数：
        model_path (str): 模型路径
    """
    net = load_model(model_path)
    env = TicTacToe()
    state = env.reset()

    print("你是 X（AI是 O），请开始你的回合：")

    while True:
        # 人类先手
        action = human_move(state)
        state, reward, done, _ = env.step(action)

        if done:
            print("你赢了！🎉") if reward == 1 else print("平局。🤝")
            break

        # AI 回应
        ai_action, win_rate = ai_move(state, net)
        print(f"AI 落子于 {ai_action}，估计胜率: {win_rate:.2f}")
        state, reward, done, _ = env.step(ai_action)

        if done:
            print("你输了 😢") if reward == -1 else print("平局。🤝")
            break

if __name__ == "__main__":
    MODEL_PATH = "/mnt/data1/zhongrongmiao/lm/tictactoe_mcts_rl/policy_value_net.pth"
    play_against_ai(MODEL_PATH)