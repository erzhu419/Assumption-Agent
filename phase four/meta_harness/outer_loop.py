"""
Outer Meta-Harness search loop.

At each iteration:
  1. Evaluate the latest harness on the search set (shared across iterations)
  2. Log score + reasoning to history
  3. Proposer reads full history and writes a new harness
  4. Validate interface (syntax + solve() signature)
  5. Loop

After N iterations, the Pareto-best harness is evaluated on the held-out set.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List

import numpy as np

PROJECT = Path(__file__).parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT.parent.parent / "phase one"))
sys.path.insert(0, str(PROJECT.parent.parent / "phase zero" / "scripts"))

import mh_config as cfg
from runtime import make_context, load_harness, load_kb
from evaluator import evaluate, log_eval
from proposer import HarnessProposer, HistoryEntry


def load_history(harnesses_dir: Path, log_path: Path) -> List[HistoryEntry]:
    """Reconstruct history from logs + harness source files."""
    if not log_path.exists():
        return []
    entries_by_harness = {}
    for line in log_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        entries_by_harness[rec["harness"]] = rec

    history: List[HistoryEntry] = []
    for harness_name, rec in sorted(entries_by_harness.items()):
        src_path = harnesses_dir / harness_name
        if not src_path.exists():
            continue
        code = src_path.read_text(encoding="utf-8")
        # Truncate source to avoid blowing proposer context
        if len(code) > 2500:
            code = code[:2500] + "\n# ... (truncated)"
        history.append(HistoryEntry(
            version=harness_name,
            source_code=code,
            win_rate=rec.get("win_rate", 0.0),
            mean_delta=rec.get("mean_delta", 0.0),
            wins=rec.get("wins", 0),
            losses=rec.get("losses", 0),
            ties=rec.get("ties", 0),
            by_domain=rec.get("by_domain", {}),
            judge_reasoning_examples=[],  # populated below from trial dumps if available
        ))
    return history


def validate_harness_code(code: str, out_path: Path) -> bool:
    """Write, import, check for solve()."""
    out_path.write_text(code, encoding="utf-8")
    try:
        fn = load_harness(out_path)
        import inspect
        sig = inspect.signature(fn)
        if len(sig.parameters) != 2:
            print(f"  [invalid] solve() takes {len(sig.parameters)} args, need 2")
            return False
        return True
    except Exception as e:
        print(f"  [invalid] {out_path.name}: {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=cfg.NUM_ITERATIONS)
    ap.add_argument("--search-size", type=int, default=cfg.SEARCH_SET_SIZE)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)

    # Fixed search set (same problems across iterations so scores are comparable)
    sys.path.insert(0, str(PROJECT.parent.parent / "phase one"))
    from task_env.base_env import TaskEnvironment
    kb = load_kb()
    env = TaskEnvironment(strategy_kb=kb)
    val_pool = env.get_all_problems("val")
    random.shuffle(val_pool)
    search_set = val_pool[: args.search_size]
    print(f"Search set size: {len(search_set)} problems")

    # Baseline
    baseline_path = cfg.HARNESSES_DIR / "v000_baseline.py"
    assert baseline_path.exists(), "baseline missing"
    log_path = cfg.LOGS_DIR / "scores.jsonl"
    cfg.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # Evaluate baseline against itself? No — proposer doesn't need baseline's score.
    # We evaluate each proposer-generated harness against v000_baseline.

    # Fresh log for this run
    if log_path.exists():
        archive = cfg.LOGS_DIR / f"scores.backup_{int(time.time())}.jsonl"
        log_path.rename(archive)
        print(f"  archived prior log to {archive.name}")

    ctx = make_context()
    proposer = HarnessProposer()

    history: List[HistoryEntry] = []

    # === First iteration: evaluate baseline against itself ===
    # Skip: it would be 50-50 by definition. Just log a stub.

    for it in range(args.iterations):
        print(f"\n{'='*60}\n  ITERATION {it+1}/{args.iterations}\n{'='*60}")

        # Propose new harness
        print(f"  [propose] reading history ({len(history)} entries)...")
        code = proposer.propose(history, it + 1)

        new_path = cfg.HARNESSES_DIR / f"v{it+1:03d}.py"
        if not validate_harness_code(code, new_path):
            print("  [fail] generated harness invalid; skipping iteration")
            continue

        # Evaluate
        print(f"  [evaluate] running {new_path.name} on {len(search_set)} problems...")
        t0 = time.time()
        result = evaluate(new_path, search_set, ctx, baseline_path, seed=args.seed + it)
        print(f"    win_rate={result.win_rate:.1%} (w/l/t={result.wins}/{result.losses}/{result.ties}), "
              f"mean_Δ={result.mean_delta:+.2f}, errors={result.errors}, "
              f"calls/solve={result.mean_harness_calls:.1f}, {time.time() - t0:.0f}s")

        # Log
        log_eval(result, log_path)
        # Dump trial details for future proposer reasoning samples
        trial_path = cfg.LOGS_DIR / f"trials_{new_path.stem}.json"
        trial_path.write_text(json.dumps([asdict(t) for t in result.trials],
                                          indent=2, ensure_ascii=False))

        # Reload history with this new entry (and populate reasoning examples)
        history = load_history(cfg.HARNESSES_DIR, log_path)
        # Populate judge_reasoning_examples from disk
        for entry in history:
            tpath = cfg.LOGS_DIR / f"trials_{Path(entry.version).stem}.json"
            if tpath.exists():
                try:
                    trials = json.loads(tpath.read_text(encoding="utf-8"))
                    entry.judge_reasoning_examples = [
                        t.get("judge_reasoning", "")[:200]
                        for t in trials[:3] if t.get("judge_reasoning")
                    ]
                except Exception:
                    pass

    # Final summary
    print(f"\n{'='*60}\n  SEARCH COMPLETE\n{'='*60}")
    history = load_history(cfg.HARNESSES_DIR, log_path)
    history.sort(key=lambda h: -h.win_rate)
    print(f"\n  Top 5 harnesses (by win rate):")
    for h in history[:5]:
        print(f"    {h.version:<14}  win_rate={h.win_rate:.1%}  mean_Δ={h.mean_delta:+.2f}  "
              f"w/l/t={h.wins}/{h.losses}/{h.ties}")

    if history:
        best = history[0]
        print(f"\n  Best: {best.version}")


if __name__ == "__main__":
    main()
