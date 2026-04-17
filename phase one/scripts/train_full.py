"""
Phase 1: Full training with DQN and SAC-Discrete (PyTorch).
Runs both algorithms and compares results.

Usage:
    python train_full.py                           # Both DQN + SAC, 10000 eps
    python train_full.py --algo dqn --episodes 5000
    python train_full.py --algo sac --episodes 5000
    python train_full.py --skip-llm-features       # Use heuristic features
"""

import sys
import json
import argparse
import random
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter, deque

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT.parent / "phase half"))

import _config as cfg


def train_one(algo: str, task_env, extractor, episodes: int,
              save_dir: Path) -> dict:
    """Train one algorithm and return results."""

    print(f"\n{'='*60}")
    print(f"Training: {algo.upper()}, {episodes} episodes")
    print(f"{'='*60}")

    # Create dispatcher
    if algo == "dqn":
        from dispatcher.torch_dispatcher import DQNDispatcher
        dispatcher = DQNDispatcher(
            input_dim=cfg.INPUT_DIM,
            num_actions=cfg.NUM_ACTIONS,
            lr=1e-3,
            epsilon_decay=episodes // 2,
            batch_size=64,
        )
    elif algo == "sac":
        from dispatcher.torch_dispatcher import SACDiscreteDispatcher
        dispatcher = SACDiscreteDispatcher(
            input_dim=cfg.INPUT_DIM,
            num_actions=cfg.NUM_ACTIONS,
            lr=3e-4,
            batch_size=64,
        )
    elif algo == "resac":
        from dispatcher.resac_discrete import RESACDiscreteDispatcher
        dispatcher = RESACDiscreteDispatcher(
            input_dim=cfg.INPUT_DIM,
            num_actions=cfg.NUM_ACTIONS,
            ensemble_size=5,
            lr=3e-4,
            beta=-1.0,       # pessimistic (LCB)
            beta_ood=0.01,
            critic_actor_ratio=2,
            batch_size=64,
        )
    else:
        raise ValueError(f"Unknown algo: {algo}")

    print(f"Parameters: {dispatcher.param_count():,}")

    # Training loop
    reward_history = deque(maxlen=500)
    selection_history = deque(maxlen=500)
    log_interval = max(episodes // 20, 100)
    t0 = time.time()

    prev_features = None

    for ep in range(episodes):
        # Sample task
        obs = task_env.sample_task("train")

        # Extract features
        features_dict = extractor.extract(obs.description, problem_id=obs.problem_id)
        feature_vec = extractor.features_to_vector(features_dict)

        # Select action
        action = dispatcher.select_action(feature_vec, cfg.ACTION_SPACE)

        # Evaluate
        outcome = task_env.evaluate_strategy_selection(
            obs.problem_id, action.strategy_id, action.confidence
        )

        # Compute reward
        from training.reward import compute_reward
        reward = compute_reward(outcome, confidence=action.confidence)

        # Store transition (use current state as both state and next_state
        # since each episode is one-step in our setting)
        next_features = feature_vec  # one-step MDP
        dispatcher.store(feature_vec, action.action_index, reward,
                        next_features, done=True)

        # Update
        loss = dispatcher.update()

        reward_history.append(reward)
        selection_history.append(action.strategy_id)

        # Log
        if (ep + 1) % log_interval == 0:
            avg_r = np.mean(list(reward_history))
            recent = list(selection_history)
            dist = Counter(recent)
            top3 = dist.most_common(3)
            top3_ratio = sum(c for _, c in top3) / len(recent)
            elapsed = time.time() - t0

            extra = ""
            if algo == "dqn":
                extra = f"ε={dispatcher.epsilon:.2f}"
            elif algo == "sac":
                extra = f"α={dispatcher.alpha:.3f}"
            elif algo == "resac":
                extra = f"α={dispatcher.alpha:.3f}, β={dispatcher.beta:.1f}, ens={dispatcher.ensemble_size}"

            print(f"  [ep {ep+1:>5}] avg_r={avg_r:.3f}, "
                  f"top3={[(s, f'{c/len(recent):.0%}') for s, c in top3]}, "
                  f"{extra}, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nTraining complete: {elapsed:.0f}s")

    # Save
    save_path = save_dir / f"dispatcher_{algo}.pt"
    dispatcher.save(str(save_path))
    print(f"Saved to {save_path}")

    # Evaluate
    dispatcher.training = False
    from evaluation.evaluate import evaluate_dispatcher as eval_fn

    # Quick eval on test set
    problems = task_env.get_all_problems("test")
    correct = 0
    total = 0
    for p in problems:
        fd = extractor.extract(p["description"], problem_id=p.get("problem_id"))
        fv = extractor.features_to_vector(fd)
        act = dispatcher.select_action(fv, cfg.ACTION_SPACE)
        ref = p.get("reference_answer", {})
        all_good = set(ref.get("optimal_strategies", []) +
                      ref.get("acceptable_strategies", []))
        if act.strategy_id in all_good:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    print(f"Test accuracy: {accuracy:.1%} ({correct}/{total})")

    return {
        "algo": algo,
        "episodes": episodes,
        "test_accuracy": accuracy,
        "avg_reward_final": float(np.mean(list(reward_history))),
        "elapsed_seconds": elapsed,
        "params": dispatcher.param_count(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["dqn", "sac", "resac", "all"], default="all")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--skip-llm-features", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    print("Loading data...")
    strategy_kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategy_kb[d["id"]] = d
    print(f"  {len(strategy_kb)} strategies")

    from task_env.base_env import TaskEnvironment
    task_env = TaskEnvironment(strategy_kb=strategy_kb)
    print(f"  {task_env.stats()}")

    from dispatcher.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(use_llm=not args.skip_llm_features)

    save_dir = PROJECT / "checkpoints"
    save_dir.mkdir(exist_ok=True)

    # Baselines
    from evaluation.evaluate import baseline_random, baseline_most_frequent
    random.seed(args.seed)
    b_random = baseline_random(task_env, "test")
    b_freq = baseline_most_frequent(task_env, "test")
    print(f"\nBaselines — Random: {b_random['accuracy']:.1%}, "
          f"Most-frequent: {b_freq['accuracy']:.1%}")

    # Train
    results = []
    algos = ["dqn", "sac", "resac"] if args.algo == "all" else [args.algo]

    for algo in algos:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        r = train_one(algo, task_env, extractor, args.episodes, save_dir)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'Accuracy':>10} {'Avg Reward':>12} {'Time':>8}")
    print(f"{'-'*50}")
    print(f"{'Random':<20} {b_random['accuracy']:>10.1%} {'—':>12} {'—':>8}")
    print(f"{'Most-frequent':<20} {b_freq['accuracy']:>10.1%} {'—':>12} {'—':>8}")
    for r in results:
        print(f"{r['algo'].upper():<20} {r['test_accuracy']:>10.1%} "
              f"{r['avg_reward_final']:>12.3f} {r['elapsed_seconds']:>7.0f}s")

    # Save
    out = {
        "baselines": {"random": b_random["accuracy"], "most_frequent": b_freq["accuracy"]},
        "results": results,
        "config": {"episodes": args.episodes, "seed": args.seed},
    }
    out_path = PROJECT / "analysis" / "full_training_results.json"
    out_path.parent.mkdir(exist_ok=True)
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
