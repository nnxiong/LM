# train.py
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
from typing import List, Tuple

from env import TicTacToe
from model import PolicyValueNet
from mcts import Node, select_leaf, backpropagate, get_action_probs

def play_game(net: PolicyValueNet) -> List[Tuple[np.ndarray, List[float], float]]:
    """
    自我对弈一局，返回轨迹数据

    参数：
        net (PolicyValueNet): 策略价值网络

    返回：
        trajectory (List[Tuple[state, probs, reward]]): 游戏轨迹
    """
    env = TicTacToe()
    state = env.reset()
    trajectory = []
    done = False

    while not done:
        root = Node(state)
        root.expand(net)
        for _ in range(50):  # MCTS 搜索次数
            leaf = select_leaf(root, net)
            value = leaf.evaluate(net)
            backpropagate(leaf, value)

        probs = get_action_probs(root)
        action = np.random.choice(9, p=probs)
        next_state, reward, done, _ = env.step(action)
        trajectory.append((state, probs, reward))
        state = next_state

    # 最后一轮反向传播奖励
    final_reward = trajectory[-1][2]
    for i in range(len(trajectory)):
        trajectory[i] = (trajectory[i][0], trajectory[i][1], final_reward)
        final_reward = -final_reward

    return trajectory


def train():
    """
    训练主循环
    """
    net = PolicyValueNet()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    net.to(device)

    buffer = deque(maxlen=1000)  # 经验回放缓冲区

    for episode in range(200):
        print(f"\nEpisode {episode}")
        trajectories = [play_game(net)]
        buffer.extend(trajectories)

        # 从经验池中采样
        batch = random.sample(buffer, min(32, len(buffer)))

        states = torch.tensor(np.array([t[0] for traj in batch for t in traj]), dtype=torch.float32).to(device)
        target_policies = torch.tensor(np.array([t[1] for traj in batch for t in traj]), dtype=torch.float32).to(device)
        target_values = torch.tensor(np.array([[t[2]] for traj in batch for t in traj]), dtype=torch.float32).to(device)

        optimizer.zero_grad()
        policy_logits, values = net(states)
        loss_policy = F.cross_entropy(policy_logits, target_policies)
        loss_value = F.mse_loss(values, target_values)
        loss = loss_policy + loss_value
        loss.backward()
        optimizer.step()

        print(f"Loss Policy: {loss_policy.item():.4f}, Loss Value: {loss_value.item():.4f}")

    # 在 train.py 的最后添加：
    torch.save(net.state_dict(), "/mnt/data1/zhongrongmiao/lm/tictactoe_mcts_rl/policy_value_net.pth")
    print("模型已保存为 policy_value_net.pth")

if __name__ == "__main__":
    train()