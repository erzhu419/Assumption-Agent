"""
Phase 1: Evaluation — run dispatcher + baselines on test set.

Baselines:
1. Random strategy selection
2. Most-frequent strategy (always pick S02)
3. LLM self-select (ask LLM directly)
4. Rule-based matching (KB conditions matching)

Usage:
    python evaluate.py                # Full evaluation
    python evaluate.py --baseline-only  # Only baselines
"""

import json
import sys
import random
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "phase zero" / "scripts"))


def evaluate_dispatcher(dispatcher, task_env, extractor, split="test") -> Dict:
    """Evaluate trained dispatcher on a dataset split."""
    from config.settings import ACTION_SPACE

    dispatcher.training = False
    problems = task_env.get_all_problems(split)

    correct_top1 = 0
    correct_top3 = 0
    total = 0
    rewards = []

    for problem in problems:
        features_dict = extractor.extract(problem["description"])
        feature_vec = extractor.features_to_vector(features_dict)
        action = dispatcher.select_action(feature_vec, ACTION_SPACE)

        ref = problem.get("reference_answer", {})
        optimal = set(ref.get("optimal_strategies", []))
        acceptable = set(ref.get("acceptable_strategies", []))
        all_good = optimal | acceptable

        if action.strategy_id in optimal:
            correct_top1 += 1
            rewards.append(1.0)
        elif action.strategy_id in acceptable:
            correct_top1 += 1
            rewards.append(0.7)
        else:
            rewards.append(0.0)

        total += 1

    dispatcher.training = True

    return {
        "split": split,
        "total": total,
        "top1_accuracy": correct_top1 / total if total > 0 else 0,
        "avg_reward": np.mean(rewards) if rewards else 0,
        "n_problems": total,
    }


def baseline_random(task_env, split="test") -> Dict:
    """Baseline 1: Random strategy selection."""
    from config.settings import STRATEGY_IDS

    problems = task_env.get_all_problems(split)
    correct = 0
    total = 0

    for problem in problems:
        selected = random.choice(STRATEGY_IDS)
        ref = problem.get("reference_answer", {})
        all_good = set(ref.get("optimal_strategies", []) +
                      ref.get("acceptable_strategies", []))
        if selected in all_good:
            correct += 1
        total += 1

    return {"baseline": "random", "accuracy": correct / total if total > 0 else 0, "n": total}


def baseline_most_frequent(task_env, split="test") -> Dict:
    """Baseline 2: Always pick the most frequently correct strategy."""
    problems = task_env.get_all_problems(split)

    # Find most frequent optimal strategy
    strategy_counts = Counter()
    for p in task_env.get_all_problems("train"):
        for s in p.get("reference_answer", {}).get("optimal_strategies", []):
            strategy_counts[s] += 1

    most_common = strategy_counts.most_common(1)[0][0] if strategy_counts else "S02"

    correct = 0
    total = 0
    for problem in problems:
        ref = problem.get("reference_answer", {})
        all_good = set(ref.get("optimal_strategies", []) +
                      ref.get("acceptable_strategies", []))
        if most_common in all_good:
            correct += 1
        total += 1

    return {"baseline": f"most_frequent ({most_common})",
            "accuracy": correct / total if total > 0 else 0, "n": total}


def run_full_evaluation(dispatcher, task_env, extractor) -> Dict:
    """Run dispatcher + all baselines."""
    print("=== Phase 1 Evaluation ===\n")

    # Dispatcher
    print("Evaluating trained dispatcher...")
    dispatcher_result = evaluate_dispatcher(dispatcher, task_env, extractor, "test")
    print(f"  Dispatcher: top1={dispatcher_result['top1_accuracy']:.1%}, "
          f"avg_reward={dispatcher_result['avg_reward']:.3f}")

    # Baselines
    random.seed(42)
    b_random = baseline_random(task_env, "test")
    print(f"  Random: {b_random['accuracy']:.1%}")

    b_freq = baseline_most_frequent(task_env, "test")
    print(f"  Most frequent: {b_freq['accuracy']:.1%}")

    results = {
        "dispatcher": dispatcher_result,
        "baselines": {
            "random": b_random,
            "most_frequent": b_freq,
        },
    }

    # Summary
    print(f"\n{'='*40}")
    d_acc = dispatcher_result["top1_accuracy"]
    r_acc = b_random["accuracy"]
    improvement = d_acc - r_acc
    print(f"Dispatcher vs Random: {d_acc:.1%} vs {r_acc:.1%} (Δ={improvement:+.1%})")

    if d_acc > r_acc * 2:
        print("✓ Dispatcher significantly outperforms random")
    elif d_acc > r_acc:
        print("~ Dispatcher slightly better than random")
    else:
        print("✗ Dispatcher not better than random — more training needed")

    return results
