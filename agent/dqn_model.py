"""
Deep Q-Network Model - Version 6.1 (Reverted to v5.0 simplicity)
Clean implementation without unnecessary complications
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Deep Q-Network Architecture - Simplified and Effective

    Based on insights from:
    - https://github.com/aome510/chrome-dino-game-rl
    - https://github.com/Lumotheninja/dino-reinforcement-learning

    v6.1: Removed all dropout layers - they were harmful for this simple task
    v7.0: State size increased to 6 (added speed + obstacle height)
    """

    def __init__(self, state_size: int = 6, action_size: int = 2):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)

        # Initialize weights properly
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN - Separates value and advantage

    Reference: https://arxiv.org/abs/1511.06581
    """

    def __init__(self, state_size: int = 6, action_size: int = 2):
        super(DuelingDQN, self).__init__()

        # Shared feature layer
        self.feature = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Value stream (how good is this state?)
        self.value = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Advantage stream (how much better is each action?)
        self.advantage = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_size)
        )

    def forward(self, x):
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)

        # Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
