"""Phase 4 v3 — Autonomous wisdom library evolution loop.

MVP (Week 1): failure-driven only. No success distill / pruner / cross-LLM yet.

Each round:
  1. Pull next batch of N problems (from test split, not in sample_100/holdout_50)
  2. Run v20 solve → collect (problem, draft, final)
  3. Judge vs baseline_long on this batch → identify failures
  4. failure_generator → 0-2 candidate wisdoms
  5. For each candidate:
     a. Add to library (versioned)
     b. Re-solve this batch with extended library
     c. Compare batch win rate: with_candidate vs without
     d. KEEP if +5pp, REVERT otherwise
  6. Log round to library_evolution.json
"""

import argparse
import json
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase four"))

from llm_client import create_client
from cached_framework import _generate_with_retry, judge_pair, BASELINE_PROMPT
from wisdom_registry import (
    load_or_init_registry, save_registry, append_wisdom,
    record_activation, export_to_wisdom_library, active_wisdoms,
    deprecate_wisdom,
)
from failure_generator import generate_candidates as gen_failure_candidates


CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
AUTO_DIR.mkdir(parents=True, exist_ok=True)

EVOLUTION_LOG = AUTO_DIR / "library_evolution.json"
BATCH_LEDGER = AUTO_DIR / "batch_ledger.json"  # tracks which problems used in which round

USED_PROBLEMS_PATH = AUTO_DIR / "used_problems.json"  # pids seen in any round
CURRENT_LIBRARY = AUTO_DIR / "current_wisdom_library.json"  # v20-compatible export


def _now():
    return datetime.now(timezone.utc).isoformat()


def cache_load(p, default=None):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default if default is not None else {}
    return default if default is not None else {}


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def pull_unused_batch(n: int, used: set) -> list:
    """Pull N unseen problems from test split.
    Excludes sample_100, sample_holdout_50, and previously-used loop problems."""
    sys.path.insert(0, str(PROJECT))
    import _config as cfg
    from task_env.base_env import TaskEnvironment

    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kb[d["id"]] = d
    env = TaskEnvironment(strategy_kb=kb)
    test_pool = env.get_all_problems("test")

    # Exclude sample_100, holdout_50, and used
    s100 = {p["problem_id"] for p in json.loads((CACHE / "sample_100.json").read_text())}
    h50 = {p["problem_id"] for p in json.loads((CACHE / "sample_holdout_50.json").read_text())}
    excluded = s100 | h50 | used

    available = [p for p in test_pool if p["problem_id"] not in excluded]
    if len(available) < n:
        print(f"  [WARN] only {len(available)} unused problems available, wanted {n}")
        return available

    rng = random.Random(len(used) + n)  # deterministic but varies with state
    rng.shuffle(available)
    return available[:n]


def run_v20_on_batch(problems: list, wisdom_path: Path) -> dict:
    """Run v20 framework on a batch. Returns {pid: answer}."""
    # Save batch as temporary sample file
    tmp_sample_name = f"_tmp_batch_{int(time.time())}.json"
    tmp_sample = CACHE / tmp_sample_name
    cache_save(tmp_sample, problems)

    variant = f"_tmp_v20_{int(time.time())}"
    try:
        cmd = [
            "python", "-u",
            str(PROJECT / "phase one" / "scripts" / "validation" / "phase2_v20_framework.py"),
            "--variant", variant,
            "--n", str(len(problems)),
        ]
        # Note: phase2_v20_framework currently hardcodes sample_100.json
        # So we need to either patch it OR save as sample_100.json (dangerous)
        # For MVP we'll do the SAFER approach: patch v20 to accept --sample
        # But for now, run with the temp file name by swapping.
        # → this is a limitation. For MVP, patch v20 later; for now use sample_100.json workaround.
        raise NotImplementedError(
            "v20 framework needs --sample / --wisdom CLI args. "
            "Patch phase2_v20_framework.py first, then call from here."
        )
    finally:
        if tmp_sample.exists():
            tmp_sample.unlink()


def run_round(registry: dict, round_num: int, batch_size: int = 20):
    """Execute one full round of the autonomous loop.

    **MVP Stub**: currently only does:
      - Pull batch
      - Log round start
      - (Real v20 invocation + A/B test TODO after v20 framework gets --wisdom CLI)
    """
    used_pids = set(cache_load(USED_PROBLEMS_PATH, default=[]))

    print(f"\n===== ROUND {round_num} =====")
    print(f"  Registry version: {registry['version']}")
    print(f"  Active wisdoms:   {len(active_wisdoms(registry))}")
    print(f"  Used problems so far: {len(used_pids)}")

    batch = pull_unused_batch(batch_size, used_pids)
    print(f"  Pulled batch: {len(batch)} problems")
    if len(batch) < 3:
        print(f"  [SKIP] insufficient problems")
        return None

    batch_pids = [p["problem_id"] for p in batch]
    used_pids.update(batch_pids)
    cache_save(USED_PROBLEMS_PATH, sorted(used_pids))

    # TODO: full loop implementation requires v20 framework to accept --wisdom
    # and --sample CLI arguments. That patch needs to happen before this stub
    # can do full generation + A/B.

    log_entry = {
        "round": round_num,
        "timestamp": _now(),
        "library_version": registry["version"],
        "active_wisdoms": len(active_wisdoms(registry)),
        "batch_pids": batch_pids,
        "status": "mvp_stub — v20 patching TODO",
    }
    return log_entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    registry = load_or_init_registry()
    evolution = cache_load(EVOLUTION_LOG, default=[])

    start_round = evolution[-1]["round"] + 1 if evolution else 1

    for i in range(args.rounds):
        round_num = start_round + i
        entry = run_round(registry, round_num, args.batch_size)
        if entry is not None:
            evolution.append(entry)
            cache_save(EVOLUTION_LOG, evolution)
            save_registry(registry)

    print(f"\n=== Autonomous loop complete ===")
    print(f"  Total rounds run: {len(evolution)}")
    print(f"  Final registry version: {registry['version']}")


if __name__ == "__main__":
    main()
