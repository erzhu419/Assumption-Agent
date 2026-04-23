"""Phase 4 v3 — Autonomous wisdom library evolution loop (Fix A+B applied).

Fixes vs MVP v1:
  A: Accumulate failures across rounds in `failure_buffer.json`. Only trigger
     candidate generation when buffer ≥ BUFFER_THRESHOLD (default 15).
  B: Add baseline_long opponent. Each round runs baseline_long on batch,
     judges v20 vs baseline_long, adds v20's losses/ties to buffer.

Each round:
  1. Pull batch of N unseen problems
  2. Run v20 (base library) + baseline_long in parallel
  3. Judge v20 vs baseline_long on batch
  4. Append v20 losses/ties to failure_buffer
  5. If buffer ≥ BUFFER_THRESHOLD:
     a. Feed buffer to failure_generator → candidates
     b. For each candidate:
        - Build extended library
        - Re-run v20 with extended library on CURRENT batch
        - A/B judge ext vs base
        - KEEP if +10pp, else REVERT
     c. On KEEP: drop buffer entries covered by new wisdom
  6. Log round
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
from cached_framework import judge_pair, _save_content_cache
from wisdom_registry import (
    load_or_init_registry, save_registry, append_wisdom,
    active_wisdoms,
)
from failure_generator import generate_candidates as gen_failure_candidates


CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
AUTO_DIR.mkdir(parents=True, exist_ok=True)

EVOLUTION_LOG = AUTO_DIR / "library_evolution.json"
USED_PROBLEMS_PATH = AUTO_DIR / "used_problems.json"
CURRENT_LIBRARY = AUTO_DIR / "current_wisdom_library.json"
FAILURE_BUFFER = AUTO_DIR / "failure_buffer.json"

V20_SCRIPT = PROJECT / "phase one" / "scripts" / "validation" / "phase2_v20_framework.py"
BASELINE_LONG_SCRIPT = PROJECT / "phase one" / "scripts" / "validation" / "phase2_baseline_long.py"

BUFFER_THRESHOLD = 15  # accumulate this many failures before triggering propose
KEEP_THRESHOLD = 0.10  # +10pp on A/B to KEEP candidate


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
    sys.path.insert(0, str(PROJECT))
    import _config as cfg
    from task_env.base_env import TaskEnvironment

    kb = {}
    for f in sorted(cfg.KB_DIR.glob("S*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        kb[d["id"]] = d
    env = TaskEnvironment(strategy_kb=kb)
    test_pool = env.get_all_problems("test")

    s100 = {p["problem_id"] for p in json.loads((CACHE / "sample_100.json").read_text())}
    h50 = {p["problem_id"] for p in json.loads((CACHE / "sample_holdout_50.json").read_text())}
    excluded = s100 | h50 | used

    available = [p for p in test_pool if p["problem_id"] not in excluded]
    if len(available) < n:
        print(f"  [WARN] only {len(available)} unused problems, wanted {n}")
        return available

    rng = random.Random(len(used) + n)
    rng.shuffle(available)
    return available[:n]


def run_subprocess(cmd: list, timeout: int = 2400) -> bool:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        print(f"    ERROR: {result.stderr[-400:]}")
        return False
    return True


def run_v20(variant: str, sample_filename: str, wisdom_filename: str, n: int) -> dict:
    cmd = [
        "python", "-u", str(V20_SCRIPT),
        "--variant", variant, "--n", str(n),
        "--sample", sample_filename,
        "--wisdom", wisdom_filename,
    ]
    print(f"    [v20 {variant}] ...")
    if not run_subprocess(cmd):
        return {}
    return cache_load(CACHE / "answers" / f"{variant}_answers.json")


def run_baseline_long(variant: str, sample_filename: str, n: int) -> dict:
    cmd = [
        "python", "-u", str(BASELINE_LONG_SCRIPT),
        "--variant", variant, "--n", str(n),
        "--sample", sample_filename,
    ]
    print(f"    [baseline_long {variant}] ...")
    if not run_subprocess(cmd):
        return {}
    return cache_load(CACHE / "answers" / f"{variant}_answers.json")


def judge_a_vs_b(batch: list, ans_a: dict, ans_b: dict, label_a: str, label_b: str) -> dict:
    client = create_client()
    wins_a = wins_b = ties = 0
    details = {}
    for p in batch:
        pid = p["problem_id"]
        a, b = ans_a.get(pid), ans_b.get(pid)
        if not a or not b:
            continue
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, a_was = a, b, "A"
        else:
            left, right, a_was = b, a, "B"
        verdict = judge_pair(client, p.get("description", ""), left, right)
        winner_raw = verdict.get("winner", "tie")
        if winner_raw == "tie":
            winner = "tie"; ties += 1
        elif winner_raw == a_was:
            winner = label_a; wins_a += 1
        else:
            winner = label_b; wins_b += 1
        details[pid] = {"winner": winner, "reasoning": verdict.get("reasoning", "")}
    _save_content_cache()
    total_decided = wins_a + wins_b
    wr_a = wins_a / total_decided if total_decided else 0.5
    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties, "wr_a": wr_a, "details": details}


def run_round(registry: dict, round_num: int, batch_size: int = 20):
    used_pids = set(cache_load(USED_PROBLEMS_PATH, default=[]))
    failure_buffer = cache_load(FAILURE_BUFFER, default=[])

    print(f"\n===== ROUND {round_num} =====")
    print(f"  Registry: {registry['version']}, active={len(active_wisdoms(registry))}")
    print(f"  Buffer size: {len(failure_buffer)}")
    print(f"  Used problems: {len(used_pids)}")

    batch = pull_unused_batch(batch_size, used_pids)
    if len(batch) < 3:
        print("  [SKIP] insufficient problems")
        return None
    batch_pids = [p["problem_id"] for p in batch]
    print(f"  Batch: {len(batch)} problems")

    # --- Step 1: export current library ---
    META_STRIP = {"status", "created_at", "last_activated", "activation_count",
                   "contribution_gain", "source", "keep_reason", "gain_samples",
                   "deprecated_at", "deprecation_reason", "removed_at", "removal_reason"}
    base_lib_export = [
        {k: v for k, v in w.items() if k not in META_STRIP}
        for w in active_wisdoms(registry)
    ]
    cache_save(CURRENT_LIBRARY, base_lib_export)
    base_lib_filename = f"_loop_r{round_num}_base_library.json"
    cache_save(CACHE / base_lib_filename, base_lib_export)

    # Save batch
    batch_filename = f"_loop_r{round_num}_batch.json"
    cache_save(CACHE / batch_filename, batch)

    # --- Step 2: run v20 + baseline_long ---
    base_variant = f"_loop_r{round_num}_v20_base"
    bl_variant = f"_loop_r{round_num}_bllong"

    print("  [1/4] Running v20 (base library)...")
    t0 = time.time()
    answers_v20 = run_v20(base_variant, batch_filename, base_lib_filename, len(batch))
    print(f"    v20 done in {time.time()-t0:.0f}s ({len(answers_v20)} answers)")

    print("  [1/4] Running baseline_long...")
    t0 = time.time()
    answers_bl = run_baseline_long(bl_variant, batch_filename, len(batch))
    print(f"    baseline_long done in {time.time()-t0:.0f}s ({len(answers_bl)} answers)")

    if not answers_v20 or not answers_bl:
        print("  [ERROR] gen failed, skipping round")
        return None

    # --- Step 3: judge v20 vs baseline_long, accumulate failures ---
    print("  [2/4] Judging v20 vs baseline_long...")
    ab = judge_a_vs_b(batch, answers_v20, answers_bl, "v20", "baseline_long")
    print(f"    v20 wins {ab['wins_a']}, baseline_long wins {ab['wins_b']}, ties {ab['ties']}")
    print(f"    v20 wr on this batch: {ab['wr_a']:.2f}")

    # Add v20 losses/ties to buffer (these are problems where baseline_long matched or beat v20)
    for p in batch:
        pid = p["problem_id"]
        d = ab["details"].get(pid)
        if not d:
            continue
        if d["winner"] != "v20":
            failure_buffer.append({
                "pid": pid,
                "round": round_num,
                "problem": p.get("description", "")[:300],
                "v20_answer": answers_v20.get(pid, "")[:400],
                "opponent_answer": answers_bl.get(pid, "")[:400],
                "judge_reason": d.get("reasoning", "")[:200],
                "judge_winner": d["winner"],
            })

    print(f"    Buffer size after round: {len(failure_buffer)}")

    actions = []
    # --- Step 4: if buffer threshold reached, trigger propose ---
    if len(failure_buffer) >= BUFFER_THRESHOLD:
        print(f"  [3/4] Buffer threshold reached ({BUFFER_THRESHOLD}). Triggering propose...")
        sample_for_info = batch + [
            {"problem_id": f["pid"], "description": f["problem"],
             "domain": "?", "difficulty": "?"}
            for f in failure_buffer if f["pid"] not in batch_pids
        ]
        candidates = gen_failure_candidates(registry, failure_buffer, sample_for_info)
        print(f"    → {len(candidates)} candidate(s)")

        # --- Step 5: A/B each candidate on CURRENT batch ---
        for cand in candidates:
            aphorism = cand.get("aphorism", "?")
            print(f"  [4/4] A/B testing '{aphorism[:30]}'...")

            ext_lib = list(base_lib_export)
            tentative_id = f"WTEST{round_num}_{len(actions)}"
            cand_entry = {**cand, "id": tentative_id}
            for meta_k in ["_source", "novelty_sim", "covers_batch_pids", "rationale"]:
                cand_entry.pop(meta_k, None)
            ext_lib.append(cand_entry)

            ext_lib_filename = f"_loop_r{round_num}_ext_{tentative_id}.json"
            cache_save(CACHE / ext_lib_filename, ext_lib)

            ext_variant = f"_loop_r{round_num}_v20_ext_{tentative_id}"
            answers_ext = run_v20(ext_variant, batch_filename, ext_lib_filename, len(batch))
            if not answers_ext:
                actions.append({"candidate": aphorism, "decision": "gen_failed"})
                continue

            ab_res = judge_a_vs_b(batch, answers_ext, answers_v20, "extended", "base")
            wr_ext = ab_res["wr_a"]
            print(f"    Result: ext wins {ab_res['wins_a']}, base wins {ab_res['wins_b']}, "
                  f"ties {ab_res['ties']}, wr_ext={wr_ext:.2f}")

            if wr_ext >= 0.5 + KEEP_THRESHOLD:
                new_id = append_wisdom(
                    registry, cand, cand.get("_source", "failure_driven"),
                    f"r{round_num} A/B wr_ext={wr_ext:.2f}"
                )
                actions.append({
                    "candidate": aphorism, "assigned_id": new_id,
                    "decision": "KEEP", "wr_ext": wr_ext,
                })
                print(f"    ✅ KEEP as {new_id}")
                # Drop buffer entries covered by this wisdom
                covered = set(cand.get("covers_batch_pids", []))
                failure_buffer = [f for f in failure_buffer if f["pid"] not in covered]
                print(f"    Buffer pruned to {len(failure_buffer)}")
            else:
                actions.append({
                    "candidate": aphorism, "decision": "REVERT",
                    "wr_ext": wr_ext, "reason": f"wr_ext<{0.5+KEEP_THRESHOLD}"
                })
                print(f"    ❌ REVERT (wr_ext {wr_ext:.2f} < {0.5+KEEP_THRESHOLD:.2f})")

    # --- Cleanup ---
    for f in [batch_filename, base_lib_filename]:
        p = CACHE / f
        if p.exists():
            p.unlink()
    # ext_lib tmp files kept only until round done (for debugging)

    used_pids.update(batch_pids)
    cache_save(USED_PROBLEMS_PATH, sorted(used_pids))
    cache_save(FAILURE_BUFFER, failure_buffer)

    entry = {
        "round": round_num,
        "timestamp": _now(),
        "library_version_end": registry["version"],
        "active_wisdoms_end": len(active_wisdoms(registry)),
        "batch_pids": batch_pids,
        "v20_wr_vs_baseline_long": ab["wr_a"],
        "buffer_size_end": len(failure_buffer),
        "candidates_proposed": len([a for a in actions if "decision" in a]),
        "actions": actions,
    }
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=20)
    args = ap.parse_args()

    registry = load_or_init_registry()
    evolution = cache_load(EVOLUTION_LOG, default=[])
    start_round = evolution[-1]["round"] + 1 if evolution else 1

    t0 = time.time()
    for i in range(args.rounds):
        round_num = start_round + i
        entry = run_round(registry, round_num, args.batch_size)
        if entry:
            evolution.append(entry)
            cache_save(EVOLUTION_LOG, evolution)
            save_registry(registry)
            print(f"  Round {round_num} done. "
                  f"Registry {registry['version']}, "
                  f"active={len(active_wisdoms(registry))}, "
                  f"buffer={entry['buffer_size_end']}")

    total = time.time() - t0
    print(f"\n=== Loop complete ===")
    print(f"  Rounds: {len(evolution)}")
    print(f"  Registry: {registry['version']} ({len(active_wisdoms(registry))} active)")
    print(f"  Elapsed: {total:.0f}s ({total/60:.1f} min)")


if __name__ == "__main__":
    main()
