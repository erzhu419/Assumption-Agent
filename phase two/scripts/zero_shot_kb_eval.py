"""
Zero-shot evaluation: pick strategy by KBMatcher argmax alone (no RL training).
Measures the intrinsic discriminative value of the KB, independent of the
dispatcher's weights. If the post-Phase2 KB scores higher than the pre-snapshot
KB under this protocol, the new conditions carry genuine information.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase one"))

import _config as cfg
from dispatcher.kb_matcher import KBMatcher  # noqa: E402


def evaluate(kb_dir: Path, problems, feature_cache: dict,
             strategy_ids, action_space) -> dict:
    matcher = KBMatcher(kb_dir, strategy_ids)

    correct = 0
    total = 0
    correct_by_difficulty = {"easy": [0, 0], "medium": [0, 0], "hard": [0, 0]}
    by_strategy = {}

    for p in problems:
        feats = feature_cache.get(p["problem_id"], {})
        if not feats:
            # fall back to unbiased
            total += 1
            continue
        scores = matcher.compute_scores(feats, action_space)
        # Argmax over strategies only (skip compositions/special for fairness)
        s_slice = scores[:len(strategy_ids)]
        if float(s_slice.max()) <= 0.5 + 1e-6:
            # No discriminative signal: skip (random-equivalent)
            pred = "NONE"
        else:
            pred = strategy_ids[int(np.argmax(s_slice))]

        ref = p.get("reference_answer", {})
        good = set(ref.get("optimal_strategies", []) +
                   ref.get("acceptable_strategies", []))
        hit = pred in good
        if hit:
            correct += 1
        total += 1

        diff = p.get("difficulty", "medium")
        if diff in correct_by_difficulty:
            correct_by_difficulty[diff][1] += 1
            if hit:
                correct_by_difficulty[diff][0] += 1

        by_strategy.setdefault(pred, 0)
        by_strategy[pred] += 1

    acc = correct / total if total else 0
    acc_by_diff = {k: (v[0] / v[1] if v[1] else 0) for k, v in correct_by_difficulty.items()}

    return {
        "kb_dir": str(kb_dir),
        "accuracy": acc,
        "correct": correct,
        "total": total,
        "accuracy_by_difficulty": acc_by_diff,
        "top_predictions": sorted(by_strategy.items(), key=lambda x: -x[1])[:10],
    }


def main():
    # Load problems (test split = shuffled slice from base_env seed=42)
    from task_env.base_env import TaskEnvironment  # noqa: E402
    import sys as _s
    _s.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))
    strategy_kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.load(open(f, encoding="utf-8"))
        strategy_kb[d["id"]] = d
    env = TaskEnvironment(strategy_kb=strategy_kb)
    problems = env.get_all_problems("test")

    # Features
    feat_cache = PROJECT.parent / "phase one" / "cache" / "features.json"
    features = json.loads(feat_cache.read_text(encoding="utf-8"))

    sys.path.insert(0, str(PROJECT.parent / "phase one"))
    import _config as p1cfg

    pre = PROJECT / "analysis" / "kb_snapshot_pre"
    post = cfg.KB_DIR

    print("Zero-shot KB evaluation on TEST set")
    print(f"{'-'*60}")
    r_pre = evaluate(pre, problems, features, p1cfg.STRATEGY_IDS, p1cfg.ACTION_SPACE)
    print(f"  KB=pre : acc={r_pre['accuracy']:.1%} ({r_pre['correct']}/{r_pre['total']})")
    print(f"             by_diff={r_pre['accuracy_by_difficulty']}")
    r_post = evaluate(post, problems, features, p1cfg.STRATEGY_IDS, p1cfg.ACTION_SPACE)
    print(f"  KB=post: acc={r_post['accuracy']:.1%} ({r_post['correct']}/{r_post['total']})")
    print(f"             by_diff={r_post['accuracy_by_difficulty']}")
    print(f"  Δ = {r_post['accuracy'] - r_pre['accuracy']:+.1%}")

    out = PROJECT / "analysis" / "zero_shot_kb_results.json"
    out.write_text(json.dumps({"pre": r_pre, "post": r_post}, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
