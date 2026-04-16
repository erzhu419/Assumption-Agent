"""
Phase 1: PyTorch-based Dispatcher with DQN and SAC-Discrete.
Replaces the numpy MLP prototype with proper RL algorithms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Tuple


@dataclass
class DispatcherAction:
    strategy_id: str
    action_index: int
    confidence: float
    log_prob: float
    value_estimate: float
    backup_strategy: Optional[str] = None


# =========================================================================
# Shared network architecture
# =========================================================================

class QNetwork(nn.Module):
    """Shared Q-network for both DQN and SAC-Discrete."""

    def __init__(self, input_dim: int, num_actions: int,
                 hidden1: int = 256, hidden2: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
        )
        self.q_head = nn.Linear(hidden2, num_actions)
        self.v_head = nn.Linear(hidden2, 1)  # for SAC-Discrete

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (q_values, state_value)."""
        h = self.net(x)
        return self.q_head(h), self.v_head(h)


# =========================================================================
# Replay Buffer
# =========================================================================

class ReplayBuffer:
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# =========================================================================
# DQN Dispatcher
# =========================================================================

class DQNDispatcher:
    """
    Double DQN with target network for discrete strategy selection.
    Simple, stable, well-suited for 33-action problems.
    """

    def __init__(self, input_dim: int, num_actions: int,
                 lr: float = 1e-3, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05,
                 epsilon_decay: int = 5000, target_update: int = 100,
                 buffer_size: int = 50000, batch_size: int = 64):
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.step_count = 0

        # Networks
        self.q_net = QNetwork(input_dim, num_actions)
        self.target_net = QNetwork(input_dim, num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.training = True

    def select_action(self, features: np.ndarray,
                      action_space: list = None) -> DispatcherAction:
        state = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            q_values, v = self.q_net(state)
            q_values = q_values.squeeze(0)
            probs = F.softmax(q_values, dim=-1).numpy()

        # Epsilon-greedy during training
        if self.training and random.random() < self.epsilon:
            action_idx = random.randrange(self.num_actions)
        else:
            action_idx = int(q_values.argmax().item())

        confidence = float(probs[action_idx])
        log_prob = float(np.log(probs[action_idx] + 1e-10))
        value = float(v.item())

        strategy_id = action_space[action_idx] if action_space else str(action_idx)

        # Backup
        sorted_idx = np.argsort(probs)[::-1]
        backup_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
        backup_id = action_space[backup_idx] if action_space else None

        return DispatcherAction(
            strategy_id=strategy_id,
            action_index=action_idx,
            confidence=confidence,
            log_prob=log_prob,
            value_estimate=value,
            backup_strategy=backup_id,
        )

    def store(self, state, action, reward, next_state, done=False):
        self.buffer.push(state, action, reward, next_state, float(done))

    def update(self) -> float:
        """One DQN update step. Returns loss."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q
        q_values, _ = self.q_net(states)
        q_current = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: use q_net to select action, target_net to evaluate
        with torch.no_grad():
            next_q, _ = self.q_net(next_states)
            next_actions = next_q.argmax(dim=1)
            target_q, _ = self.target_net(next_states)
            q_target = rewards + self.gamma * (1 - dones) * target_q.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

        loss = F.mse_loss(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # Update epsilon
        self.step_count += 1
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon - (1.0 - self.epsilon_end) / self.epsilon_decay,
        )

        # Update target network
        if self.step_count % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "epsilon": self.epsilon,
        }, path)

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.q_net.load_state_dict(data["q_net"])
        self.target_net.load_state_dict(data["target_net"])
        self.optimizer.load_state_dict(data["optimizer"])
        self.step_count = data["step_count"]
        self.epsilon = data["epsilon"]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.q_net.parameters())


# =========================================================================
# SAC-Discrete Dispatcher
# =========================================================================

class SACDiscreteDispatcher:
    """
    Soft Actor-Critic for discrete actions.
    Better sample efficiency than DQN, entropy-regularized for exploration.

    Reference: Christodoulou (2019) "Soft Actor-Critic for Discrete Action Settings"
    """

    def __init__(self, input_dim: int, num_actions: int,
                 lr: float = 3e-4, gamma: float = 0.99,
                 tau: float = 0.005, alpha: float = 0.2,
                 auto_alpha: bool = True,
                 buffer_size: int = 50000, batch_size: int = 64):
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Networks: two Q-networks + target (twin Q for stability)
        self.q1 = QNetwork(input_dim, num_actions)
        self.q2 = QNetwork(input_dim, num_actions)
        self.q1_target = QNetwork(input_dim, num_actions)
        self.q2_target = QNetwork(input_dim, num_actions)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Policy network (separate from Q for SAC)
        self.policy = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Auto temperature tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -np.log(1.0 / num_actions) * 0.5
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        self.buffer = ReplayBuffer(buffer_size)
        self.training = True
        self.step_count = 0

    def _get_action_probs(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get action probabilities and log probs from policy."""
        logits = self.policy(state)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs

    def select_action(self, features: np.ndarray,
                      action_space: list = None) -> DispatcherAction:
        state = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            probs, log_probs = self._get_action_probs(state)
            probs_np = probs.squeeze(0).numpy()

            q1_vals, _ = self.q1(state)
            value = (probs * q1_vals).sum().item()

        if self.training:
            action_idx = int(torch.multinomial(probs.squeeze(0), 1).item())
        else:
            action_idx = int(probs.argmax(dim=-1).item())

        confidence = float(probs_np[action_idx])
        log_prob = float(log_probs.squeeze(0)[action_idx].item())

        strategy_id = action_space[action_idx] if action_space else str(action_idx)

        sorted_idx = np.argsort(probs_np)[::-1]
        backup_idx = sorted_idx[1] if len(sorted_idx) > 1 else sorted_idx[0]
        backup_id = action_space[backup_idx] if action_space else None

        return DispatcherAction(
            strategy_id=strategy_id,
            action_index=action_idx,
            confidence=confidence,
            log_prob=log_prob,
            value_estimate=value,
            backup_strategy=backup_id,
        )

    def store(self, state, action, reward, next_state, done=False):
        self.buffer.push(state, action, reward, next_state, float(done))

    def update(self) -> float:
        """One SAC-Discrete update step."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # --- Q-function update ---
        with torch.no_grad():
            next_probs, next_log_probs = self._get_action_probs(next_states)
            next_q1, _ = self.q1_target(next_states)
            next_q2, _ = self.q2_target(next_states)
            next_q = torch.min(next_q1, next_q2)
            # V(s') = E_a[Q(s',a) - α log π(a|s')]
            next_v = (next_probs * (next_q - self.alpha * next_log_probs)).sum(dim=1)
            q_target = rewards + self.gamma * (1 - dones) * next_v

        q1_vals, _ = self.q1(states)
        q2_vals, _ = self.q2(states)
        q1_current = q1_vals.gather(1, actions.unsqueeze(1)).squeeze(1)
        q2_current = q2_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        q1_loss = F.mse_loss(q1_current, q_target)
        q2_loss = F.mse_loss(q2_current, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # --- Policy update ---
        probs, log_probs = self._get_action_probs(states)
        with torch.no_grad():
            q1_vals, _ = self.q1(states)
            q2_vals, _ = self.q2(states)
            q_min = torch.min(q1_vals, q2_vals)

        # π loss = E[α log π - Q]
        policy_loss = (probs * (self.alpha * log_probs - q_min)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # --- Alpha (temperature) update ---
        if self.auto_alpha:
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            with torch.no_grad():
                self.log_alpha.clamp_(-5.0, 2.0)
            self.alpha = self.log_alpha.exp().item()

        # --- Soft target update ---
        self.step_count += 1
        for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return (q1_loss.item() + q2_loss.item()) / 2

    def save(self, path: str):
        torch.save({
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "policy": self.policy.state_dict(),
            "step_count": self.step_count,
            "alpha": self.alpha,
        }, path)

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.q1.load_state_dict(data["q1"])
        self.q2.load_state_dict(data["q2"])
        self.policy.load_state_dict(data["policy"])
        self.step_count = data["step_count"]
        self.alpha = data["alpha"]

    def param_count(self) -> int:
        return (sum(p.numel() for p in self.q1.parameters()) +
                sum(p.numel() for p in self.q2.parameters()) +
                sum(p.numel() for p in self.policy.parameters()))
