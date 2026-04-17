"""
Closed-loop validation: train RE-SAC with KB-match features, on either the
pre-Phase2 snapshot OR the current (post-Phase2) KB.

Usage:
    python retrain_with_kb.py --kb pre   --episodes 50000
    python retrain_with_kb.py --kb post  --episodes 50000
    python retrain_with_kb.py --kb both  --episodes 50000      # runs both, compares
"""

import argparse
import json
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT.parent / "phase half"))

import _config as cfg


def train_one_kb(kb_label: str, kb_dir: Path, episodes: int, seed: int,
                 task_env, extractor, feature_cache: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Training RE-SAC on KB={kb_label}  ({kb_dir})")
    print(f"{'='*60}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    from dispatcher.resac_discrete import RESACDiscreteDispatcher
    from dispatcher.kb_matcher import KBMatcher
    from training.reward import compute_reward

    matcher = KBMatcher(kb_dir, cfg.STRATEGY_IDS)

    dispatcher = RESACDiscreteDispatcher(
        input_dim=cfg.INPUT_DIM, num_actions=cfg.NUM_ACTIONS,
        ensemble_size=5, lr=3e-4, beta=-1.0, beta_ood=0.01,
        critic_actor_ratio=2, batch_size=64,
    )

    reward_history = deque(maxlen=500)
    selection_history = deque(maxlen=500)
    log_interval = max(episodes // 20, 100)
    t0 = time.time()

    for ep in range(episodes):
        obs = task_env.sample_task("train")
        features_dict = extractor.extract(obs.description, problem_id=obs.problem_id)

        # Structural features from precompute cache
        cached_struct = feature_cache.get(obs.problem_id, {})
        if cached_struct:
            features_dict.update(cached_struct)

        kb_scores = matcher.compute_scores(features_dict, cfg.ACTION_SPACE)
        vec = extractor.features_to_vector(features_dict, kb_match_scores=kb_scores)

        action = dispatcher.select_action(vec, cfg.ACTION_SPACE)
        outcome = task_env.evaluate_strategy_selection(
            obs.problem_id, action.strategy_id, action.confidence
        )
        reward = compute_reward(outcome, confidence=action.confidence)
        dispatcher.store(vec, action.action_index, reward, vec, done=True)
        dispatcher.update()

        reward_history.append(reward)
        selection_history.append(action.strategy_id)

        if (ep + 1) % log_interval == 0:
            avg_r = np.mean(list(reward_history))
            recent = list(selection_history)
            top3 = Counter(recent).most_common(3)
            elapsed = time.time() - t0
            print(f"  [ep {ep+1:>5}] avg_r={avg_r:.3f}, "
                  f"top3={[(s, f'{c/len(recent):.0%}') for s, c in top3]}, "
                  f"α={dispatcher.alpha:.2f}, {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nTraining complete: {elapsed:.0f}s")

    # Evaluate
    dispatcher.training = False
    problems = task_env.get_all_problems("test")
    correct = 0
    total = 0
    confusion = Counter()
    for p in problems:
        fd = extractor.extract(p["description"], problem_id=p.get("problem_id"))
        cached_struct = feature_cache.get(p["problem_id"], {})
        if cached_struct:
            fd.update(cached_struct)
        kb_scores = matcher.compute_scores(fd, cfg.ACTION_SPACE)
        fv = extractor.features_to_vector(fd, kb_match_scores=kb_scores)
        act = dispatcher.select_action(fv, cfg.ACTION_SPACE)
        ref = p.get("reference_answer", {})
        all_good = set(ref.get("optimal_strategies", []) +
                       ref.get("acceptable_strategies", []))
        if act.strategy_id in all_good:
            correct += 1
        total += 1
        confusion[act.strategy_id] += 1

    accuracy = correct / total if total else 0
    print(f"Test accuracy (KB={kb_label}): {accuracy:.1%} ({correct}/{total})")

    return {
        "kb_label": kb_label,
        "episodes": episodes,
        "test_accuracy": accuracy,
        "avg_reward_final": float(np.mean(list(reward_history))),
        "elapsed_seconds": elapsed,
        "top10_strategies": Counter(confusion).most_common(10),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", choices=["pre", "post", "both"], default="both")
    ap.add_argument("--episodes", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data + feature cache once
    print("Loading data...")
    strategy_kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategy_kb[d["id"]] = d

    from task_env.base_env import TaskEnvironment
    task_env = TaskEnvironment(strategy_kb=strategy_kb)
    print(f"  {task_env.stats()}")

    from dispatcher.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(use_llm=False)

    feature_cache_path = PROJECT / "cache" / "features.json"
    feature_cache = {}
    if feature_cache_path.exists():
        feature_cache = json.loads(feature_cache_path.read_text(encoding="utf-8"))

    # KB paths
    pre_dir = PROJECT.parent / "phase two" / "analysis" / "kb_snapshot_pre"
    post_dir = cfg.KB_DIR

    labels = []
    if args.kb in ("pre", "both"):
        labels.append(("pre", pre_dir))
    if args.kb in ("post", "both"):
        labels.append(("post", post_dir))

    results = []
    for lbl, kb in labels:
        r = train_one_kb(lbl, kb, args.episodes, args.seed,
                         task_env, extractor, feature_cache)
        results.append(r)

    print(f"\n{'='*60}")
    print("KB CLOSED-LOOP COMPARISON")
    print(f"{'='*60}")
    for r in results:
        print(f"  KB={r['kb_label']:>4}: test_acc={r['test_accuracy']:.1%}, "
              f"avg_r={r['avg_reward_final']:.3f}, {r['elapsed_seconds']:.0f}s")
    if len(results) == 2:
        delta = results[1]["test_accuracy"] - results[0]["test_accuracy"]
        print(f"\n  Δ test_acc (post - pre) = {delta:+.1%}")
        print(f"  {'KB evolution HELPS' if delta > 0 else 'KB evolution does NOT help'} the dispatcher")

    out = PROJECT.parent / "phase two" / "analysis" / "closed_loop_results.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
