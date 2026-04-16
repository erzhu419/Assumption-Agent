"""
Phase 1: Reward function for dispatcher training.
Includes confidence calibration (Brier score style).
"""

import numpy as np


def compute_reward(outcome, confidence: float = 0.5,
                   weights: dict = None) -> float:
    """
    Compute reward for a strategy selection.

    Args:
        outcome: ExecutionOutcome with success, evaluation_score, consistency_score
        confidence: dispatcher's confidence in its selection (0-1)
        weights: override default reward weights
    """
    from _config import REWARD_WEIGHTS
    w = weights or REWARD_WEIGHTS

    # Completion reward
    r_completion = outcome.evaluation_score

    # Consistency (did executor follow the strategy?)
    r_consistency = outcome.consistency_score

    # Efficiency (fewer steps = better, but we don't have real step data yet)
    r_efficiency = max(0, 1.0 - outcome.steps_taken / 10.0)

    # Progress (partial success counts)
    r_progress = 0.5 if outcome.partial_success else (1.0 if outcome.success else 0.0)

    # Step progress (placeholder — needs real trajectory data)
    r_step_progress = r_progress  # Same as progress for now

    # Confidence calibration (Brier score style)
    actual = 1.0 if outcome.success else 0.0
    calibration_error = (confidence - actual) ** 2
    r_calibration = 1.0 - calibration_error

    reward = (
        w["completion"] * r_completion +
        w["consistency"] * r_consistency +
        w["efficiency"] * r_efficiency +
        w["progress"] * r_progress +
        w["step_progress"] * r_step_progress +
        w["calibration"] * r_calibration
    )

    return float(np.clip(reward, 0.0, 1.0))


def compute_composition_reward(step_outcomes: list, weights: dict = None) -> float:
    """Reward for strategy compositions (COMP_*)."""
    if not step_outcomes:
        return 0.0

    # Final step determines main reward
    final = step_outcomes[-1]
    base = compute_reward(final, weights=weights)

    # Transition efficiency
    n_transitions = len(step_outcomes) - 1
    if n_transitions > 0:
        smooth = sum(1 for o in step_outcomes[:-1] if o.success or o.partial_success)
        transition_factor = smooth / n_transitions
    else:
        transition_factor = 1.0

    # Slight discount for using composition (prefer simple single strategy)
    return base * transition_factor * 0.95
