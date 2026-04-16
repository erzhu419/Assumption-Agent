"""
Phase 1: PPO Training Loop for MLP Dispatcher.
Integrates with Phase 0.5 world model for model-based RL.
"""

import numpy as np
import json
import time
import random
from pathlib import Path
from collections import Counter, deque
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Episode:
    features: np.ndarray
    action_idx: int
    log_prob: float
    value: float
    reward: float
    confidence: float
    is_real: bool = True


@dataclass
class TrainingStats:
    episode: int = 0
    total_reward: float = 0.0
    avg_reward_100: float = 0.0
    top1_accuracy: float = 0.0
    strategy_distribution: Dict = field(default_factory=dict)
    collapse_detected: bool = False
    world_model_tau: float = 0.0


class PPOTrainer:
    """
    Train MLP dispatcher with PPO.

    Features:
    - Model-based RL (90% simulated, 10% real via world model)
    - Curriculum learning (easy → medium → hard)
    - Strategy collapse monitoring
    - Early stopping
    """

    def __init__(self, dispatcher, task_env, feature_extractor,
                 world_model=None, config: dict = None):
        from _config import PPO_CONFIG, MODEL_BASED_CONFIG, COLLAPSE_CONFIG, ACTION_SPACE

        self.dispatcher = dispatcher
        self.task_env = task_env
        self.extractor = feature_extractor
        self.world_model = world_model
        self.ppo_config = config or PPO_CONFIG
        self.mb_config = MODEL_BASED_CONFIG
        self.collapse_config = COLLAPSE_CONFIG
        self.action_space = ACTION_SPACE

        # Training buffers
        self.episode_buffer: List[Episode] = []
        self.reward_history: deque = deque(maxlen=1000)
        self.selection_history: deque = deque(maxlen=self.collapse_config["window"])

        # Stats
        self.stats = TrainingStats()
        self.best_val_accuracy = 0.0
        self.patience_counter = 0

    def train(self, max_episodes: int = None) -> TrainingStats:
        """Main training loop."""
        max_ep = max_episodes or self.ppo_config["max_episodes"]
        batch_size = self.ppo_config["batch_size"]

        print(f"Training dispatcher: {max_ep} episodes, batch={batch_size}")
        print(f"Action space: {len(self.action_space)} actions")
        print(f"Params: {self.dispatcher.param_count():,}")

        t0 = time.time()

        for ep in range(max_ep):
            self.stats.episode = ep

            # --- Decide: real or simulated ---
            use_real = self._should_use_real(ep)

            # --- Sample task ---
            obs = self.task_env.sample_task("train")

            # --- Extract features ---
            features_dict = self.extractor.extract(obs.description)
            feature_vec = self.extractor.features_to_vector(features_dict)

            # --- Select action ---
            action = self.dispatcher.select_action(feature_vec, self.action_space)

            # --- Execute and get reward ---
            if use_real:
                outcome = self.task_env.evaluate_strategy_selection(
                    obs.problem_id, action.strategy_id, action.confidence
                )
                # Update world model with real result
                if self.world_model:
                    self.world_model.update(
                        features_dict, action.strategy_id,
                        outcome.success,
                        selector_confidence=action.confidence,
                        strategy_consistency=outcome.consistency_score,
                    )
            else:
                # Use world model simulation
                if self.world_model:
                    sim = self.world_model.simulate_execution(
                        features_dict, action.strategy_id,
                    )
                    from task_env.base_env import ExecutionOutcome
                    outcome = ExecutionOutcome(
                        success=sim["success"],
                        partial_success=sim["partial_success"],
                        evaluation_score=sim["evaluation_score"],
                        strategy_used=action.strategy_id,
                        consistency_score=0.8,
                        is_simulated=True,
                    )
                else:
                    # No world model: must use real
                    outcome = self.task_env.evaluate_strategy_selection(
                        obs.problem_id, action.strategy_id, action.confidence
                    )

            # --- Compute reward ---
            from training.reward import compute_reward
            reward = compute_reward(outcome, confidence=action.confidence)

            # --- Store episode ---
            self.episode_buffer.append(Episode(
                features=feature_vec,
                action_idx=action.action_index,
                log_prob=action.log_prob,
                value=action.value_estimate,
                reward=reward,
                confidence=action.confidence,
                is_real=use_real or self.world_model is None,
            ))

            self.reward_history.append(reward)
            self.selection_history.append(action.strategy_id)

            # --- PPO update ---
            if len(self.episode_buffer) >= batch_size:
                self._ppo_update()
                self.episode_buffer.clear()

            # --- Monitoring ---
            if (ep + 1) % 100 == 0:
                self._log_progress(ep, t0)

            # --- Collapse check ---
            if (ep + 1) % self.collapse_config["window"] == 0:
                self._check_collapse()

            # --- Early stopping ---
            if (ep + 1) % 1000 == 0 and ep >= self.ppo_config["early_stopping_min_episodes"]:
                if self._check_early_stop():
                    print(f"Early stopping at episode {ep+1}")
                    break

        elapsed = time.time() - t0
        print(f"\nTraining complete: {self.stats.episode+1} episodes, {elapsed:.0f}s")
        return self.stats

    def _should_use_real(self, episode: int) -> bool:
        """Decide whether to use real or simulated execution."""
        if self.world_model is None:
            return True

        if episode < self.mb_config["cold_start_episodes"]:
            return random.random() < self.mb_config["cold_start_real_ratio"]
        else:
            return random.random() < self.mb_config["real_ratio"]

    def _ppo_update(self):
        """PPO parameter update using collected episodes."""
        if not self.episode_buffer:
            return

        cfg = self.ppo_config
        eps = cfg["clip_epsilon"]
        lr = cfg["learning_rate"]

        # Compute advantages (simple: reward - value baseline)
        rewards = np.array([e.reward for e in self.episode_buffer])
        values = np.array([e.value for e in self.episode_buffer])
        advantages = rewards - values
        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(cfg["num_epochs_per_update"]):
            for i, ep in enumerate(self.episode_buffer):
                # Forward pass
                probs, new_value = self.dispatcher.forward(ep.features)
                new_log_prob = np.log(probs[ep.action_idx] + 1e-10)

                # PPO clipped objective
                ratio = np.exp(new_log_prob - ep.log_prob)
                adv = advantages[i]
                clipped_ratio = np.clip(ratio, 1 - eps, 1 + eps)
                policy_loss = -min(ratio * adv, clipped_ratio * adv)

                # Value loss
                value_loss = 0.5 * (new_value - ep.reward) ** 2

                # Entropy bonus
                entropy = -np.sum(probs * np.log(probs + 1e-10))
                entropy_loss = -self.dispatcher.entropy_coef * entropy

                # Total loss
                total_loss = policy_loss + cfg["value_loss_coef"] * value_loss + entropy_loss

                # Simple gradient update (numerical gradient approximation)
                # For a production system, use autograd (PyTorch). For prototype, this works.
                self._numerical_gradient_update(ep.features, ep.action_idx,
                                                 adv, ep.reward, lr)

    def _numerical_gradient_update(self, features, action_idx, advantage, target_value, lr):
        """
        Simple parameter update via policy gradient.
        Production would use PyTorch autograd. This is a prototype.
        """
        probs, value = self.dispatcher.forward(features)

        # Policy gradient: increase prob of good actions, decrease bad
        # ∇log π(a|s) * A(s,a)
        grad_logits = -probs.copy()
        grad_logits[action_idx] += 1  # ∂log π / ∂logit for softmax
        grad_logits *= advantage * lr

        # Apply to policy head
        h2 = np.maximum(features @ self.dispatcher.W1 + self.dispatcher.b1, 0)
        h2 = np.maximum(h2 @ self.dispatcher.W2 + self.dispatcher.b2, 0)

        self.dispatcher.W_policy += np.outer(h2, grad_logits)
        self.dispatcher.b_policy += grad_logits

        # Value update
        value_error = target_value - value
        value_grad = value_error * lr * 0.5
        self.dispatcher.W_value += h2.reshape(-1, 1) * value_grad
        self.dispatcher.b_value += value_grad

    def _log_progress(self, ep, t0):
        elapsed = time.time() - t0
        avg_r = np.mean(list(self.reward_history)[-100:]) if self.reward_history else 0
        self.stats.avg_reward_100 = avg_r

        # Strategy distribution
        recent = list(self.selection_history)
        dist = Counter(recent)
        top3 = dist.most_common(3)

        print(f"  [ep {ep+1}] avg_r={avg_r:.3f}, "
              f"top3={[(s, f'{c/len(recent):.0%}') for s, c in top3]}, "
              f"elapsed={elapsed:.0f}s")

    def _check_collapse(self):
        """Check if dispatcher is collapsing to a few strategies."""
        recent = list(self.selection_history)
        if len(recent) < self.collapse_config["window"]:
            return

        dist = Counter(recent)
        top3_ratio = sum(c for _, c in dist.most_common(3)) / len(recent)

        if top3_ratio > self.collapse_config["threshold"]:
            self.stats.collapse_detected = True
            self.dispatcher.entropy_coef *= self.collapse_config["entropy_boost"]
            print(f"  ⚠ Strategy collapse detected (top3={top3_ratio:.0%}), "
                  f"boosting entropy to {self.dispatcher.entropy_coef:.4f}")

    def _check_early_stop(self) -> bool:
        """Check if training should stop early."""
        # Evaluate on validation set
        self.dispatcher.training = False
        correct = 0
        total = 0

        for problem in self.task_env.get_all_problems("val")[:50]:
            features_dict = self.extractor.extract(problem["description"])
            feature_vec = self.extractor.features_to_vector(features_dict)
            action = self.dispatcher.select_action(feature_vec, self.action_space)

            ref = problem.get("reference_answer", {})
            optimal = set(ref.get("optimal_strategies", []) +
                         ref.get("acceptable_strategies", []))
            if action.strategy_id in optimal:
                correct += 1
            total += 1

        self.dispatcher.training = True

        accuracy = correct / total if total > 0 else 0
        improvement = accuracy - self.best_val_accuracy

        if improvement > self.ppo_config["early_stopping_threshold"]:
            self.best_val_accuracy = accuracy
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        self.stats.top1_accuracy = accuracy
        print(f"  Val accuracy: {accuracy:.1%} (best={self.best_val_accuracy:.1%}, "
              f"patience={self.patience_counter})")

        return self.patience_counter >= 3  # 3 consecutive no-improvement checks
