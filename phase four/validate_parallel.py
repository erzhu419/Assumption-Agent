"""Parallel A/B validation of all success-distilled candidates.

Constraint: speed up WITHOUT weakening the evaluation.
  - Keep full 50-problem holdout (statistical power preserved)
  - Keep +10pp KEEP threshold (decision rigor preserved)
  - Parallelize v20 subprocesses (up to PARALLEL_V20 concurrent)
  - Parallelize judges within each A/B (up to PARALLEL_JUDGES)

Expected wall time for 11 candidates + base on 50 problems:
  ~(12 tasks / 6 parallel) * ~50min/task ≈ 100 min wall
  (vs ~6 hours sequential).
"""

import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase four"))

from llm_client import create_client, parse_json_from_llm
from gpt5_client import GPT5Client
from cached_framework import judge_pair, _save_content_cache
from wisdom_registry import load_or_init_registry, save_registry, append_wisdom


CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
CANDIDATES_PATH = AUTO_DIR / "success_distilled_candidates.json"
VALIDATION_LOG = AUTO_DIR / "validation_log_parallel.json"

V20_SCRIPT = PROJECT / "phase one" / "scripts" / "validation" / "phase2_v20_framework.py"
EXEMPLARS_PATH = CACHE / "wisdom_diverse_exemplars.json"

HOLDOUT_SAMPLE = "sample_holdout_50.json"
TEST_N = 50              # full holdout — no statistical-power compromise
PARALLEL_V20 = 6         # v20 subprocesses in parallel
PARALLEL_JUDGES = 8      # judge threads in parallel per A/B
KEEP_THRESHOLD = 0.10    # +10pp — same rigor as sequential baseline


EXEMPLAR_PROMPT = """给下面 wisdom 从 50 个候选问题里挑 3 个**跨域最远**的判例。

## Wisdom
aphorism: {aphorism}
source: {source}
signal: {signal}
unpacked: {unpacked}

## 候选 problems (50 个)
{problems_brief}

## 规则
1. 挑 3 个 pid
2. 跨 2-3 个不同 domain
3. wisdom 真能在那题 fire

## 输出 JSON (不要代码块)
{{"selected": [
  {{"pid": "xxx", "why_applies": "20-40字"}},
  ... (3 条)
]}}
"""


def cache_load(p, default=None):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return default if default is not None else {}
    return default if default is not None else {}


def cache_save(p, obj):
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2))


def load_full_holdout():
    full = json.loads((CACHE / HOLDOUT_SAMPLE).read_text(encoding="utf-8"))
    return [p for p in full if "description" in p]


def mine_exemplars_for_candidate(cand, sample, v13_ans, ours_ans, client):
    pid_to_info = {p["problem_id"]: p for p in sample}
    problems_brief = "\n".join(
        f"[{p['problem_id']}] [{p.get('domain','?')}] {p.get('description','')[:80]}"
        for p in sample
    )
    prompt = EXEMPLAR_PROMPT.format(
        aphorism=cand["aphorism"], source=cand["source"],
        signal=cand["signal"], unpacked=cand["unpacked_for_llm"],
        problems_brief=problems_brief,
    )
    try:
        resp = client.generate(prompt, max_tokens=500, temperature=0.3)
        parsed = parse_json_from_llm(resp["text"])
        selected = parsed.get("selected", [])
    except Exception as e:
        print(f"    [exemplar err for {cand['aphorism'][:20]}] {e}")
        return []

    result = []
    MATH_SCI = {"mathematics", "science"}
    for item in selected[:3]:
        pid = item.get("pid", "").strip()
        if pid not in pid_to_info:
            continue
        info = pid_to_info[pid]
        dom = info.get("domain", "?")
        ans = ours_ans.get(pid) if dom in MATH_SCI else v13_ans.get(pid)
        ans = ans or v13_ans.get(pid) or ours_ans.get(pid) or ""
        result.append({
            "pid": pid, "domain": dom,
            "difficulty": info.get("difficulty", "?"),
            "problem_sketch": info.get("description", "")[:350],
            "why_applies": item.get("why_applies", ""),
            "answer_snippet": ans[:700],
            "answer_source": "ours_27" if dom in MATH_SCI else "v13_reflect",
        })
    return result


def run_v20_subprocess(variant, sample_file, wisdom_file, n):
    cmd = [
        "python", "-u", str(V20_SCRIPT),
        "--variant", variant, "--n", str(n),
        "--sample", sample_file,
        "--wisdom", wisdom_file,
    ]
    start = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start
    if result.returncode != 0:
        print(f"    [{variant} FAILED] {result.stderr[-200:]}")
        return None, elapsed
    answers = cache_load(CACHE / "answers" / f"{variant}_answers.json")
    return answers, elapsed


def run_v20_parallel(tasks):
    """tasks = [(variant, sample_file, wisdom_file, n), ...]
    Returns {variant: (answers, elapsed)}"""
    results = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_V20) as ex:
        futures = {
            ex.submit(run_v20_subprocess, *task): task[0]
            for task in tasks
        }
        for f in as_completed(futures):
            variant = futures[f]
            try:
                answers, elapsed = f.result()
                results[variant] = (answers, elapsed)
                n = len(answers) if answers else 0
                print(f"  [DONE] {variant}: {n} answers in {elapsed:.0f}s")
            except Exception as e:
                print(f"  [ERROR] {variant}: {e}")
                results[variant] = (None, 0)
    return results


def judge_pair_threaded(p, a, b, label_a, label_b):
    """Single judge call, returns (pid, winner_label, reasoning)."""
    pid = p["problem_id"]
    rng = random.Random(hash(pid) % (2**32))
    if rng.random() < 0.5:
        left, right, a_was = a, b, "A"
    else:
        left, right, a_was = b, a, "B"
    client = create_client()
    v = judge_pair(client, p.get("description", ""), left, right)
    w = v.get("winner", "tie")
    if w == "tie":
        return pid, "tie", v.get("reasoning", "")
    return (pid, label_a if w == a_was else label_b, v.get("reasoning", ""))


def judge_ab_parallel(batch, ans_a, ans_b, label_a, label_b):
    """A/B judge on batch, parallel."""
    tasks = []
    for p in batch:
        pid = p["problem_id"]
        a, b = ans_a.get(pid), ans_b.get(pid)
        if not a or not b:
            continue
        tasks.append((p, a, b))

    wins_a = wins_b = ties = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_JUDGES) as ex:
        futures = [ex.submit(judge_pair_threaded, p, a, b, label_a, label_b)
                   for p, a, b in tasks]
        for f in as_completed(futures):
            try:
                pid, winner, _ = f.result()
                if winner == "tie":
                    ties += 1
                elif winner == label_a:
                    wins_a += 1
                else:
                    wins_b += 1
            except Exception as e:
                print(f"    [judge err] {e}")
    _save_content_cache()
    total = wins_a + wins_b
    wr_a = wins_a / total if total else 0.5
    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties, "wr_a": wr_a}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-k", type=int, default=11,
                    help="number of candidates to validate")
    args = ap.parse_args()

    # Prepare candidates
    candidates = cache_load(CANDIDATES_PATH, default=[])
    seen = set()
    dedup = []
    for c in candidates:
        if c["aphorism"] in seen:
            continue
        seen.add(c["aphorism"])
        dedup.append(c)
    dedup.sort(key=lambda c: -c.get("_cluster_size", 0))
    top = dedup[:args.top_k]
    print(f"Loaded {len(candidates)}, dedup {len(dedup)}, testing {len(top)}")
    for c in top:
        print(f"  • {c['aphorism']} (n={c.get('_cluster_size','?')})")

    subset = load_full_holdout()
    if len(subset) < TEST_N:
        print(f"[WARN] holdout has only {len(subset)} problems, using all")
    print(f"\nTest set: {len(subset)} problems (full holdout_50)")

    # Load context
    registry = load_or_init_registry()
    v13_ans = cache_load(CACHE / "answers" / "phase2_v13_reflect_answers.json")
    ours_ans = cache_load(CACHE / "answers" / "ours_27_answers.json")
    META_STRIP = {"status", "created_at", "last_activated", "activation_count",
                   "contribution_gain", "source", "keep_reason", "gain_samples",
                   "deprecated_at", "deprecation_reason", "removed_at", "removal_reason"}
    base_export = [{k: v for k, v in w.items() if k not in META_STRIP}
                   for w in registry["wisdoms"] if w.get("status") == "active"]

    base_lib_filename = "_valp_base_library.json"
    cache_save(CACHE / base_lib_filename, base_export)

    # Mine exemplars for all candidates (sequential, 1 call each)
    print(f"\n[1/4] Mining exemplars for {len(top)} candidates (sequential)...")
    exemplars_all = cache_load(EXEMPLARS_PATH, default={})
    gpt_client = GPT5Client()
    t0 = time.time()
    valid_candidates = []
    for i, cand in enumerate(top):
        tentative_id = f"WCAND{i+1:02d}"
        if tentative_id in exemplars_all and len(exemplars_all[tentative_id]) == 3:
            print(f"  [{tentative_id}] reusing cached exemplars")
        else:
            ex = mine_exemplars_for_candidate(cand, subset, v13_ans, ours_ans, gpt_client)
            if len(ex) != 3:
                print(f"  [{tentative_id} {cand['aphorism'][:20]}] SKIP (only {len(ex)} exemplars)")
                continue
            exemplars_all[tentative_id] = ex
        cand["tentative_id"] = tentative_id
        valid_candidates.append(cand)
    cache_save(EXEMPLARS_PATH, exemplars_all)
    print(f"  Done in {time.time()-t0:.0f}s; {len(valid_candidates)} valid candidates")

    # Build ext libraries + prepare v20 run tasks (base + all ext)
    v20_tasks = []
    base_variant = "_valp_v20_base"
    v20_tasks.append((base_variant, HOLDOUT_SAMPLE, base_lib_filename, len(subset)))
    for cand in valid_candidates:
        tid = cand["tentative_id"]
        cand_entry = {k: v for k, v in cand.items()
                      if k not in {"_cluster_id", "_cluster_size", "_source",
                                    "novelty_sim", "covers_batch_pids", "rationale",
                                    "tentative_id"}}
        cand_entry["id"] = tid
        ext_lib = base_export + [cand_entry]
        ext_lib_filename = f"_valp_ext_{tid}.json"
        cache_save(CACHE / ext_lib_filename, ext_lib)
        ext_variant = f"_valp_v20_ext_{tid}"
        v20_tasks.append((ext_variant, HOLDOUT_SAMPLE, ext_lib_filename, len(subset)))

    print(f"\n[2/4] Running {len(v20_tasks)} v20 subprocesses in parallel "
          f"(up to {PARALLEL_V20} at a time)...")
    t0 = time.time()
    v20_results = run_v20_parallel(v20_tasks)
    print(f"  All v20 done in {time.time()-t0:.0f}s")

    # A/B each candidate vs base
    base_ans = v20_results.get(base_variant, (None, 0))[0]
    if not base_ans:
        print("  [FATAL] base v20 failed, aborting")
        return
    print(f"  base: {len(base_ans)} answers")

    print(f"\n[3/4] Running {len(valid_candidates)} A/B judges...")
    results = []
    for cand in valid_candidates:
        tid = cand["tentative_id"]
        ext_variant = f"_valp_v20_ext_{tid}"
        ext_ans, _ = v20_results.get(ext_variant, (None, 0))
        if not ext_ans:
            print(f"  [{tid}] no ext answers, SKIP")
            continue
        t0 = time.time()
        ab = judge_ab_parallel(subset, ext_ans, base_ans, "ext", "base")
        print(f"  [{tid} {cand['aphorism'][:25]}] ext={ab['wins_a']} "
              f"base={ab['wins_b']} ties={ab['ties']} wr_ext={ab['wr_a']:.2f} "
              f"({time.time()-t0:.0f}s)")

        decision = "KEEP" if ab["wr_a"] >= 0.5 + KEEP_THRESHOLD else "REVERT"
        results.append({
            "tid": tid,
            "candidate": cand["aphorism"],
            "source": cand.get("source"),
            "cluster_size": cand.get("_cluster_size"),
            "novelty_sim": cand.get("novelty_sim"),
            "ab": ab,
            "decision": decision,
        })

    # Commit KEEP candidates
    print(f"\n[4/4] Committing KEEP candidates to registry...")
    kept = []
    for r in results:
        if r["decision"] != "KEEP":
            continue
        cand = next(c for c in valid_candidates if c["tentative_id"] == r["tid"])
        cand_entry = {k: v for k, v in cand.items()
                      if k not in {"_cluster_id", "_cluster_size", "_source",
                                    "novelty_sim", "covers_batch_pids", "rationale",
                                    "tentative_id"}}
        new_id = append_wisdom(
            registry, cand_entry, "success_distilled",
            f"parallel validation wr_ext={r['ab']['wr_a']:.2f} on {TEST_N} holdout"
        )
        r["committed_id"] = new_id
        exemplars_all[new_id] = exemplars_all.pop(r["tid"], [])
        kept.append(r)
    cache_save(EXEMPLARS_PATH, exemplars_all)
    save_registry(registry)

    # Log
    log = cache_load(VALIDATION_LOG, default=[])
    log.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "test_n": TEST_N,
        "results": results,
    })
    cache_save(VALIDATION_LOG, log)

    print(f"\n=== FINAL SUMMARY ===")
    print(f"  Tested: {len(results)}")
    print(f"  KEPT:   {len(kept)}")
    for r in kept:
        print(f"    ✅ {r['candidate']} → {r.get('committed_id')} "
              f"(wr_ext={r['ab']['wr_a']:.2f})")
    reverted = [r for r in results if r["decision"] == "REVERT"]
    print(f"  REVERTED: {len(reverted)}")
    for r in reverted:
        print(f"    ❌ {r['candidate']} (wr_ext={r['ab']['wr_a']:.2f})")


if __name__ == "__main__":
    main()
