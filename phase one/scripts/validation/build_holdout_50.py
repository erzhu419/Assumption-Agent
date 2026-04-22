"""
Build sample_holdout_50.json — 50 problems NOT in sample_100.json.

Same test pool, different seed, excluded overlap. Used for Phase A validation:
does v12c's +14pp vs v11 hold on unseen problems?
"""

import json
import random
import sys
from pathlib import Path

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent / "phase zero" / "scripts"))

import _config as cfg

CACHE = PROJECT.parent / "phase two" / "analysis" / "cache"
OUT_PATH = CACHE / "sample_holdout_50.json"


def main():
    # Load existing 100-sample to avoid overlap
    existing = json.loads((CACHE / "sample_100.json").read_text(encoding="utf-8"))
    existing_ids = {p["problem_id"] for p in existing}
    print(f"Excluding {len(existing_ids)} problems already in sample_100")

    # Load full test pool
    from task_env.base_env import TaskEnvironment
    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kb[d["id"]] = d
    env = TaskEnvironment(strategy_kb=kb)
    test_pool = env.get_all_problems("test")
    print(f"Test pool size: {len(test_pool)}")

    # Filter out overlap
    available = [p for p in test_pool if p["problem_id"] not in existing_ids]
    print(f"Available held-out pool: {len(available)}")

    if len(available) < 50:
        print(f"WARNING: only {len(available)} available")
        sample = available
    else:
        rng = random.Random(7)
        rng.shuffle(available)
        sample = available[:50]

    sample[0]["_seed"] = 7
    OUT_PATH.write_text(json.dumps(sample, ensure_ascii=False, indent=2))
    print(f"Saved {len(sample)} to {OUT_PATH.name}")

    # Domain/diff breakdown
    from collections import Counter
    dom_c = Counter(p.get("domain", "?") for p in sample)
    diff_c = Counter(p.get("difficulty", "?") for p in sample)
    print("\n  By domain:", dict(dom_c.most_common()))
    print("  By difficulty:", dict(diff_c.most_common()))


if __name__ == "__main__":
    main()
