"""
Phase 1: MLP Dispatcher (Plan A).
Small neural network that selects strategies from the knowledge base.
Trained with PPO. ~300K parameters.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class DispatcherAction:
    strategy_id: str
    action_index: int
    confidence: float
    log_prob: float
    value_estimate: float
    backup_strategy: Optional[str] = None


class MLPDispatcher:
    """
    Lightweight MLP dispatcher for strategy selection.
    Uses numpy for portability (no PyTorch/TF dependency).

    Architecture:
        Input (INPUT_DIM) → Linear(256) + ReLU
                          → Linear(128) + ReLU + Dropout
                          → Policy head (NUM_ACTIONS) + Value head (1)
    """

    def __init__(self, input_dim: int, num_actions: int, hidden1: int = 256,
                 hidden2: int = 128, lr: float = 1e-4):
        self.input_dim = input_dim
        self.num_actions = num_actions
        self.lr = lr

        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden1).astype(np.float32) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1, dtype=np.float32)
        self.W2 = np.random.randn(hidden1, hidden2).astype(np.float32) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2, dtype=np.float32)

        # Policy head
        self.W_policy = np.random.randn(hidden2, num_actions).astype(np.float32) * 0.01
        self.b_policy = np.zeros(num_actions, dtype=np.float32)

        # Value head
        self.W_value = np.random.randn(hidden2, 1).astype(np.float32) * 0.01
        self.b_value = np.zeros(1, dtype=np.float32)

        # Training state
        self.training = True
        self.entropy_coef = 0.05

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Forward pass.
        Returns (action_probs, value_estimate).
        """
        # Layer 1
        h1 = x @ self.W1 + self.b1
        h1 = np.maximum(h1, 0)  # ReLU

        # Layer 2
        h2 = h1 @ self.W2 + self.b2
        h2 = np.maximum(h2, 0)  # ReLU

        # Policy head (softmax)
        logits = h2 @ self.W_policy + self.b_policy
        logits -= logits.max()  # numerical stability
        exp_logits = np.exp(logits)
        probs = exp_logits / (exp_logits.sum() + 1e-10)

        # Value head
        value = float((h2 @ self.W_value + self.b_value)[0])

        return probs, value

    def select_action(self, features: np.ndarray,
                      action_space: list = None) -> DispatcherAction:
        """
        Select a strategy given feature vector.
        During training: sample from distribution.
        During eval: take argmax.
        """
        probs, value = self.forward(features)

        if self.training:
            # Sample
            action_idx = np.random.choice(len(probs), p=probs)
        else:
            action_idx = int(np.argmax(probs))

        confidence = float(probs[action_idx])
        log_prob = float(np.log(probs[action_idx] + 1e-10))

        # Map index to strategy ID
        if action_space:
            strategy_id = action_space[action_idx]
        else:
            from _config import IDX_TO_ACTION
            strategy_id = IDX_TO_ACTION[action_idx]

        # Backup: second-highest probability strategy
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

    def get_params(self) -> dict:
        """Get all parameters as a dict (for saving)."""
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
            "W_policy": self.W_policy, "b_policy": self.b_policy,
            "W_value": self.W_value, "b_value": self.b_value,
        }

    def set_params(self, params: dict):
        """Load parameters from dict."""
        for key, val in params.items():
            setattr(self, key, val)

    def save(self, path: str):
        np.savez(path, **self.get_params())

    def load(self, path: str):
        data = np.load(path)
        self.set_params({k: data[k] for k in data.files})

    def param_count(self) -> int:
        return sum(p.size for p in self.get_params().values())
