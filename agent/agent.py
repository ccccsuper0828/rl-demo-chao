"""
DQN Agent - Version 6.0
With Double DQN, Prioritized Experience Replay, and anti-forgetting mechanisms

References:
- Double DQN: https://arxiv.org/abs/1509.06461
- Prioritized Experience Replay: https://arxiv.org/abs/1511.05952
- Hugging Face DRL Course: https://huggingface.co/learn/deep-rl-course/en/unit3/deep-q-algorithm
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Optional

from .dqn_model import DQN
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer


class DQNAgent:
    """
    Double DQN Agent with Prioritized Experience Replay

    v6.0 improvements to prevent catastrophic forgetting:
    1. Prioritized Experience Replay - sample important transitions more often
    2. Larger replay buffer - keep good experiences longer
    3. Slower epsilon decay - maintain exploration
    4. Soft target updates - more stable learning
    """

    def __init__(
        self,
        state_size: int = 6,
        action_size: int = 2,
        learning_rate: float = 0.001,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 50000,       # v6.0: increased from 10000
        batch_size: int = 32,
        target_update_freq: int = 100,
        use_double_dqn: bool = True,
        use_per: bool = False,          # v6.0: Prioritized Experience Replay
        per_alpha: float = 0.6,         # PER priority exponent
        per_beta_start: float = 0.4,    # PER importance sampling start
        soft_update: bool = False,      # v6.0: soft target update option
        tau: float = 0.005,             # soft update rate
        device: str = None
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_per = use_per
        self.per_beta = per_beta_start
        self.soft_update = soft_update
        self.tau = tau

        # Device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")
        print(f"Double DQN: {use_double_dqn}")
        print(f"Prioritized Replay: {use_per}")
        print(f"Buffer size: {buffer_size}")

        # Networks - smaller architecture
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay buffer - choose based on use_per
        if use_per:
            self.memory = PrioritizedReplayBuffer(buffer_size, alpha=per_alpha)
        else:
            self.memory = ReplayBuffer(buffer_size)

        self.steps = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy action selection"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self) -> Optional[float]:
        """
        Perform one training step with Double DQN and optional PER

        Double DQN key insight:
        - Standard DQN: uses target network for both selecting AND evaluating actions
        - Double DQN: uses policy network to SELECT best action,
                      target network to EVALUATE that action
        This reduces overestimation bias.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch (different for PER vs standard)
        if self.use_per:
            result = self.memory.sample(self.batch_size, self.per_beta)
            if result is None:
                return None
            states, actions, rewards, next_states, dones, indices, weights = result
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            weights = None

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: policy net selects action, target net evaluates
                next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            else:
                # Standard DQN
                next_q = self.target_net(next_states).max(1)[0]

            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute TD errors for PER
        td_errors = torch.abs(current_q.squeeze() - target_q).detach()

        # Compute loss (weighted for PER)
        if self.use_per and weights is not None:
            # Weighted loss for importance sampling
            element_wise_loss = (current_q.squeeze() - target_q) ** 2
            loss = (weights * element_wise_loss).mean()

            # Update priorities in replay buffer
            priorities = td_errors.cpu().numpy() + 1e-6
            self.memory.update_priorities(indices, priorities)
        else:
            loss = nn.SmoothL1Loss()(current_q.squeeze(), target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.steps += 1
        if self.soft_update:
            # Soft update: θ_target = τ*θ_policy + (1-τ)*θ_target
            self._soft_update_target()
        elif self.steps % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def _soft_update_target(self):
        """Soft update target network parameters"""
        for target_param, policy_param in zip(
            self.target_net.parameters(), self.policy_net.parameters()
        ):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def increase_per_beta(self, increment: float = 0.001):
        """Increase PER beta towards 1.0 over training"""
        self.per_beta = min(1.0, self.per_beta + increment)

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """Save model to file"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Model loaded from {filepath}")

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]
