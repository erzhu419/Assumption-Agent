"""
OOD validation: leave-one-domain-out.

For each of the 6 domains, train RE-SAC on the other 5 and evaluate on the
held-out domain. Each LOO run is repeated with KB=pre (no derived_from hints)
and KB=v2 (36 LLM-written tentative conditions) to measure whether the KB
provides generalization value that the RL policy can't derive on its own.

Usage:
    python ood_retrain.py --episodes 15000 --domains daily_life mathematics
    python ood_retrain.py --episodes 20000          # all 6 domains
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

import _config as cfg


ALL_DOMAINS = ["business", "daily_life", "engineering", "mathematics",
               "science", "software_engineering"]


def _split_by_domain(problems, held_out: str):
    train, test = [], []
    for p in problems:
        (test if p.get("domain") == held_out else train).append(p)
    return train, test


def train_one(kb_label, kb_dir, train_problems, test_problems, episodes, seed,
              extractor, feature_cache, task_env_for_eval):
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

    # Lightweight evaluation: pick task from train set, reference its annotation
    t0 = time.time()
    reward_hist = deque(maxlen=500)
    for ep in range(episodes):
        p = random.choice(train_problems)
        task_env_for_eval._current_task = p
        feats = extractor.extract(p.get("description", ""),
                                  problem_id=p.get("problem_id"))
        cached = feature_cache.get(p["problem_id"], {})
        if cached:
            feats.update(cached)
        kb_scores = matcher.compute_scores(feats, cfg.ACTION_SPACE)
        vec = extractor.features_to_vector(feats, kb_match_scores=kb_scores)

        action = dispatcher.select_action(vec, cfg.ACTION_SPACE)
        outcome = task_env_for_eval.evaluate_strategy_selection(
            p["problem_id"], action.strategy_id, action.confidence
        )
        reward = compute_reward(outcome, confidence=action.confidence)
        dispatcher.store(vec, action.action_index, reward, vec, done=True)
        dispatcher.update()
        reward_hist.append(reward)

    elapsed = time.time() - t0

    # Evaluate on held-out domain
    dispatcher.training = False
    correct = 0
    correct_hard = [0, 0]
    for p in test_problems:
        task_env_for_eval._current_task = p
        feats = extractor.extract(p.get("description", ""),
                                  problem_id=p.get("problem_id"))
        cached = feature_cache.get(p["problem_id"], {})
        if cached:
            feats.update(cached)
        kb_scores = matcher.compute_scores(feats, cfg.ACTION_SPACE)
        vec = extractor.features_to_vector(feats, kb_match_scores=kb_scores)
        act = dispatcher.select_action(vec, cfg.ACTION_SPACE)
        ref = p.get("reference_answer", {})
        good = set(ref.get("optimal_strategies", []) +
                   ref.get("acceptable_strategies", []))
        if act.strategy_id in good:
            correct += 1
        if p.get("difficulty") == "hard":
            correct_hard[1] += 1
            if act.strategy_id in good:
                correct_hard[0] += 1

    acc = correct / max(len(test_problems), 1)
    hard_acc = correct_hard[0] / max(correct_hard[1], 1)
    return {
        "kb": kb_label,
        "train_size": len(train_problems),
        "test_size": len(test_problems),
        "accuracy": acc,
        "hard_accuracy": hard_acc,
        "avg_reward_final": float(np.mean(list(reward_hist))),
        "elapsed_seconds": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=15000)
    ap.add_argument("--domains", nargs="+", default=ALL_DOMAINS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-pre", action="store_true",
                    help="only test v2-KB (skip pre-KB baseline)")
    args = ap.parse_args()

    # Load everything
    print("Loading data...")
    strategy_kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategy_kb[d["id"]] = d

    from task_env.base_env import TaskEnvironment
    env = TaskEnvironment(strategy_kb=strategy_kb)

    from dispatcher.feature_extractor import FeatureExtractor
    extractor = FeatureExtractor(use_llm=False)

    feat_path = PROJECT / "cache" / "features.json"
    feat_cache = json.loads(feat_path.read_text(encoding="utf-8"))

    # All problems (use everything, since we partition by domain)
    all_problems = env._problems

    pre_kb = PROJECT.parent / "phase two" / "analysis" / "kb_snapshot_pre"
    post_kb = cfg.KB_DIR

    results = []
    for domain in args.domains:
        print(f"\n{'='*60}")
        print(f"Held-out: {domain}")
        print(f"{'='*60}")
        train_probs, test_probs = _split_by_domain(all_problems, domain)
        print(f"  train={len(train_probs)} / test={len(test_probs)}")

        if not args.skip_pre:
            print(f"\n  [pre-KB]  training {args.episodes} eps...")
            r_pre = train_one("pre", pre_kb, train_probs, test_probs, args.episodes,
                              args.seed, extractor, feat_cache, env)
            print(f"    acc={r_pre['accuracy']:.1%}, hard={r_pre['hard_accuracy']:.1%}, "
                  f"t={r_pre['elapsed_seconds']:.0f}s")
            r_pre["domain"] = domain
            results.append(r_pre)

        print(f"\n  [v2-KB]   training {args.episodes} eps...")
        r_post = train_one("v2", post_kb, train_probs, test_probs, args.episodes,
                           args.seed, extractor, feat_cache, env)
        print(f"    acc={r_post['accuracy']:.1%}, hard={r_post['hard_accuracy']:.1%}, "
              f"t={r_post['elapsed_seconds']:.0f}s")
        r_post["domain"] = domain
        results.append(r_post)

    # Summary
    print(f"\n{'='*60}")
    print("OOD SUMMARY (domain held out → test accuracy)")
    print(f"{'='*60}")
    print(f"{'domain':<22} {'pre':>7} {'v2':>7} {'Δ':>7}  {'hard_pre':>10} {'hard_v2':>9}")
    print("-" * 72)
    grouped = {}
    for r in results:
        grouped.setdefault(r["domain"], {})[r["kb"]] = r
    pos, neg = 0, 0
    for dom in args.domains:
        pre_r = grouped.get(dom, {}).get("pre")
        v2_r = grouped.get(dom, {}).get("v2")
        if not v2_r:
            continue
        pre_acc = pre_r["accuracy"] if pre_r else float("nan")
        v2_acc = v2_r["accuracy"]
        delta = v2_acc - pre_acc if pre_r else float("nan")
        hp = pre_r["hard_accuracy"] if pre_r else float("nan")
        hv = v2_r["hard_accuracy"]
        if pre_r and delta > 0:
            pos += 1
        elif pre_r and delta < 0:
            neg += 1
        print(f"{dom:<22} {pre_acc:>7.1%} {v2_acc:>7.1%} {delta:>+7.1%}  "
              f"{hp:>10.1%} {hv:>9.1%}")

    print("-" * 72)
    if not args.skip_pre:
        print(f"  wins (v2 better): {pos}/{len(args.domains)}, "
              f"losses: {neg}/{len(args.domains)}")

    out = PROJECT.parent / "phase two" / "analysis" / "ood_results.json"
    out.write_text(json.dumps({
        "episodes": args.episodes,
        "domains": args.domains,
        "results": results,
    }, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
