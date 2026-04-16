"""
Phase 1: RE-SAC Discrete — Regularized Ensemble SAC for discrete action spaces.

Adapted from your RE-SAC (continuous) implementation:
- VectorizedLinear ensemble Q-networks (parallel computation)
- Epistemic uncertainty: policy uses q_mean + beta * q_std
- OOD regularization: penalizes cross-ensemble Q-value disagreement
- Auto-alpha (entropy temperature tuning)

Key change from continuous RE-SAC:
- Q-network: input = obs only, output = Q(s, a) for all actions (no action concat)
- Policy: Categorical (softmax) instead of TanhGaussian
- No reparameterization trick needed (discrete = direct expectation)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DispatcherAction:
    strategy_id: str
    action_index: int
    confidence: float
    log_prob: float
    value_estimate: float
    backup_strategy: Optional[str] = None


# =========================================================================
# Vectorized Ensemble Linear (from your RE-SAC)
# =========================================================================

class VectorizedLinear(nn.Module):
    """Batched linear layer for ensemble computation.
    Processes all ensemble members in parallel via a single batched matmul.
    """

    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for i in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (ensemble_size, batch_size, in_features)
        return x @ self.weight + self.bias


# =========================================================================
# Vectorized Ensemble Q-Network (discrete version)
# =========================================================================

class EnsembleQNetwork(nn.Module):
    """
    Vectorized ensemble of Q-networks for DISCRETE actions.

    Difference from continuous RE-SAC:
    - Input: obs only (not obs+action)
    - Output: Q(s, a) for all actions simultaneously
    """

    def __init__(self, obs_dim: int, num_actions: int,
                 hidden_dim: int = 256, ensemble_size: int = 5):
        super().__init__()
        self.ensemble_size = ensemble_size
        self.num_actions = num_actions

        self.net = nn.Sequential(
            VectorizedLinear(obs_dim, hidden_dim, ensemble_size),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, ensemble_size),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, num_actions, ensemble_size),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (batch_size, obs_dim)
        Returns:
            q_values: (ensemble_size, batch_size, num_actions)
        """
        # Expand obs for all ensemble members
        # (batch_size, obs_dim) → (ensemble_size, batch_size, obs_dim)
        x = obs.unsqueeze(0).expand(self.ensemble_size, -1, -1)
        return self.net(x)


# =========================================================================
# Categorical Policy (discrete version of TanhGaussian)
# =========================================================================

class CategoricalPolicy(nn.Module):
    """Discrete policy network: obs → action probabilities."""

    def __init__(self, obs_dim: int, num_actions: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (action_probs, log_probs)."""
        logits = self.net(obs)
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        return probs, log_probs


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
# RE-SAC Discrete Dispatcher
# =========================================================================

class RESACDiscreteDispatcher:
    """
    Regularized Ensemble SAC for discrete action selection.

    From your RE-SAC paper, adapted for discrete actions:
    - N ensemble Q-networks (vectorized, not twin-Q)
    - Epistemic uncertainty: policy optimizes E[Q_mean + beta * Q_std]
    - OOD regularization: penalizes cross-ensemble disagreement
    - Auto-alpha entropy tuning

    Args:
        ensemble_size: number of Q-network ensemble members (default 5)
        beta: weight of epistemic uncertainty in policy loss
              negative = pessimistic (LCB), positive = optimistic (UCB)
        beta_ood: weight of OOD regularization (Q-std penalty on critic)
    """

    def __init__(self, input_dim: int, num_actions: int,
                 ensemble_size: int = 5,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 beta: float = -1.0,         # pessimistic by default
                 beta_ood: float = 0.01,
                 auto_alpha: bool = True,
                 alpha: float = 0.2,
                 critic_actor_ratio: int = 2,
                 buffer_size: int = 50000,
                 batch_size: int = 64):

        self.num_actions = num_actions
        self.ensemble_size = ensemble_size
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.beta_ood = beta_ood
        self.critic_actor_ratio = critic_actor_ratio
        self.batch_size = batch_size

        # Ensemble Q-networks
        self.qf = EnsembleQNetwork(input_dim, num_actions, hidden_dim, ensemble_size)
        self.target_qf = EnsembleQNetwork(input_dim, num_actions, hidden_dim, ensemble_size)
        self.target_qf.load_state_dict(self.qf.state_dict())

        # Policy
        self.policy = CategoricalPolicy(input_dim, num_actions, hidden_dim)

        # Optimizers
        self.qf_optimizer = optim.Adam(self.qf.parameters(), lr=lr)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # Auto alpha
        self.auto_alpha = auto_alpha
        if auto_alpha:
            # Target entropy = 50% of max entropy (log(num_actions))
            # 0.98 was too high — forced near-uniform distribution, alpha diverged
            self.target_entropy = -np.log(1.0 / num_actions) * 0.5
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha

        self.buffer = ReplayBuffer(buffer_size)
        self.training = True
        self.step_count = 0
        self.actor_update_count = 0

    def select_action(self, features: np.ndarray,
                      action_space: list = None) -> DispatcherAction:
        state = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            probs, log_probs = self.policy(state)
            probs_np = probs.squeeze(0).numpy()

            # Ensemble Q for value estimate
            q_ensemble = self.qf(state)  # (ensemble, 1, num_actions)
            q_mean = q_ensemble.mean(dim=0).squeeze(0)
            value = (probs.squeeze(0) * q_mean).sum().item()

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
        """One RE-SAC update step."""
        if len(self.buffer) < self.batch_size:
            return 0.0

        total_loss = 0.0

        # Higher critic-to-actor ratio (from RE-SAC)
        for critic_step in range(self.critic_actor_ratio):
            states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

            # === Critic update ===
            with torch.no_grad():
                next_probs, next_log_probs = self.policy(next_states)
                # Target ensemble Q-values
                target_q = self.target_qf(next_states)  # (ensemble, batch, actions)
                target_q_mean = target_q.mean(dim=0)     # (batch, actions)
                # V(s') = E_a[Q_mean(s',a) - α log π(a|s')]
                next_v = (next_probs * (target_q_mean - self.alpha * next_log_probs)).sum(dim=1)
                q_target = rewards + self.gamma * (1 - dones) * next_v

            # Current ensemble Q-values
            q_ensemble = self.qf(states)  # (ensemble, batch, actions)
            # Extract Q for taken actions: (ensemble, batch)
            q_current = q_ensemble.gather(
                2, actions.unsqueeze(0).unsqueeze(-1).expand(self.ensemble_size, -1, 1)
            ).squeeze(-1)

            # Ensemble loss: MSE for each member
            critic_loss = ((q_current - q_target.unsqueeze(0).expand_as(q_current)) ** 2).mean()

            # OOD regularization: penalize Q-std across ensemble
            q_std = q_ensemble.std(dim=0)  # (batch, actions)
            ood_loss = self.beta_ood * q_std.mean()

            total_critic_loss = critic_loss + ood_loss

            self.qf_optimizer.zero_grad()
            total_critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qf.parameters(), 1.0)
            self.qf_optimizer.step()

            total_loss += critic_loss.item()

        # === Policy update (once per critic_actor_ratio) ===
        probs, log_probs = self.policy(states)

        with torch.no_grad():
            q_ensemble = self.qf(states)  # (ensemble, batch, actions)
            q_mean = q_ensemble.mean(dim=0)  # (batch, actions)
            q_std = q_ensemble.std(dim=0)    # (batch, actions)

        # RE-SAC key: policy optimizes Q_mean + beta * Q_std
        # beta < 0 → pessimistic (LCB), avoids overestimation
        q_adjusted = q_mean + self.beta * q_std

        # Policy loss: minimize E_a[α log π(a|s) - Q_adjusted(s,a)]
        policy_loss = (probs * (self.alpha * log_probs - q_adjusted)).sum(dim=1).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        self.actor_update_count += 1

        # === Alpha update ===
        if self.auto_alpha:
            entropy = -(probs.detach() * log_probs.detach()).sum(dim=1).mean()
            alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # === Soft target update ===
        self.step_count += 1
        for p, tp in zip(self.qf.parameters(), self.target_qf.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return total_loss / self.critic_actor_ratio

    def save(self, path: str):
        torch.save({
            "qf": self.qf.state_dict(),
            "target_qf": self.target_qf.state_dict(),
            "policy": self.policy.state_dict(),
            "step_count": self.step_count,
            "alpha": self.alpha,
        }, path)

    def load(self, path: str):
        data = torch.load(path, weights_only=False)
        self.qf.load_state_dict(data["qf"])
        self.target_qf.load_state_dict(data["target_qf"])
        self.policy.load_state_dict(data["policy"])
        self.step_count = data["step_count"]
        self.alpha = data["alpha"]

    def param_count(self) -> int:
        return (sum(p.numel() for p in self.qf.parameters()) +
                sum(p.numel() for p in self.policy.parameters()))
