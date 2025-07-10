# model.py
from typing import Tuple
import torch
import torch.nn as nn
class PolicyValueNet(nn.Module):
    def __init__(self):
        """
        简化的策略价值网络，输出两个头：
        - policy_logits: 每个动作的 logit 分数
        - value: 当前状态的价值估计 [-1, 1]
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 128),
            nn.SELU(),
            nn.Linear(128, 64),
            nn.SELU()
        )
        self.policy = nn.Linear(64, 9)   # 动作概率分布
        self.value = nn.Linear(64, 1)    # 价值估计

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        参数：
            x (Tensor): 输入状态（batch_size x 9）

        返回：
            policy_logits (Tensor): 每个动作的 logit
            value (Tensor): 状态价值 [-1, 1]
        """
        h = self.net(x)
        policy_logits = self.policy(h)
        value = self.value(h).tanh()  # 映射到 [-1, 1]
        return policy_logits, value

