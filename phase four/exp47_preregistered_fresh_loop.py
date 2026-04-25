"""Exp 47 — Preregistered audit on a fresh 30-problem split.

Closes review weakness #1 (post-hoc audit) at strongest available
form: apply the audit-stack thresholds frozen in PREREGISTRATION_v2.md
to a problem split that's disjoint from all previously-used samples
(sample_100 / sample_holdout_50 / sample_extend_50).

Frozen protocol from PREREGISTRATION_v2.md:
  - Inner-loop gate: wr_ext >= 0.60 (the original +10pp threshold)
  - L1 cross-family threshold: wr_ext >= 0.55 (PREREG_v2 specifies
    0.55, deliberately laxer than inner-loop 0.60 to give the
    candidate a fair shot)
  - Judge family: gemini-3-flash inner, claude-haiku for L1
  - Sample size: 30 fresh pids (smaller than original 50;
    PREREG_v2 says n_inner can be smaller for replication runs)

12 candidates from validation_log_parallel.json: WCAND01-11 and
WCROSSL01.

Outcome categories pre-specified per PREREGISTRATION_v2.md:
  1. Strong replication: >= 50% of inner-loop-positive candidates
     also pass L1 on fresh data
  2. Strong null replication: 0% pass L1
  3. Partial: 1-49% pass L1
  4. Inverted: candidates that failed inner-loop pass L1

We commit to publishing whichever outcome we observe.

Cost: 12 × 30 × 2 (base+ext) = 720 generations + 720 inner judges
+ up to 12 × 30 = 360 L1 judges = ~1800 calls. ~$25, ~2h.
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))


def _load_api_keys():
    if os.environ.get("RUOLI_GPT_KEY") and os.environ.get("RUOLI_BASE_URL"):
        return
    keyfile = Path.home() / ".api_keys"
    if not keyfile.exists():
        return
    pat = re.compile(r'^\s*export\s+(\w+)=("([^"]*)"|\'([^\']*)\'|(\S+))')
    for line in keyfile.read_text().splitlines():
        m = pat.match(line)
        if not m: continue
        name = m.group(1)
        val = m.group(3) if m.group(3) is not None else (
              m.group(4) if m.group(4) is not None else m.group(5))
        os.environ.setdefault(name, val)
        if name == "RUOLI_BASE_URL":
            base = val + "/v1" if not val.endswith("/v1") else val
            os.environ.setdefault("CLAUDE_PROXY_BASE_URL", base)
            os.environ.setdefault("GPT5_BASE_URL", base)
            os.environ.setdefault("GEMINI_PROXY_BASE_URL", base)
        if name == "RUOLI_GEMINI_KEY":
            os.environ.setdefault("GEMINI_PROXY_API_KEY", val)
        if name == "RUOLI_GPT_KEY":
            os.environ.setdefault("GPT5_API_KEY", val)
        if name == "RUOLI_CLAUDE_KEY":
            os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)

_load_api_keys()

from model_router import cheap
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp47_preregistered_fresh_loop_log.json"

PARALLEL = 6
N_FRESH = 30
SEED = 2026

# Pre-specified thresholds (from PREREGISTRATION_v2.md)
INNER_THRESHOLD = 0.60   # +10pp above 0.50 baseline
L1_THRESHOLD    = 0.55   # PREREG specifies 0.55 for L1 (laxer)


FRAME_PROMPT = """对下面问题产生 frame + 重写。

## 原题
{problem}

## 输出 JSON
{{"frame": "object_level/paradigm/hybrid",
  "critical_reframe": "30-80字",
  "rewritten_problem": "120-250字"}}
"""

EXECUTE_PROMPT = """# 解决问题

## PRIMARY FRAME
- frame: {frame}
- critical reframe: {critical_reframe}

## 问题（重写）
{rewritten_problem}

## 次要参考 wisdom
{wisdom_block}

## 要求：≤ 500 字
开始："""

JUDGE_PROMPT = """方法论评审.

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

Output JSON: {{"winner": "A"/"B"/"tie", "score_a": 1-10, "score_b": 1-10,
  "reasoning": "80 chars max"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def load_full_pool():
    """Load all 1768 problems."""
    pool = []
    for f in (PROJECT / "phase zero" / "benchmark" / "problems").glob("*.json"):
        for q in json.loads(f.read_text()):
            if q.get("description"):
                pool.append(q)
    return pool


def fresh_sample():
    """Pick N_FRESH pids disjoint from sample_100/holdout_50/extend_50."""
    pool = load_full_pool()
    used = set()
    for fn in ["sample_100.json", "sample_holdout_50.json", "sample_extend_50.json"]:
        f = CACHE / fn
        if f.exists():
            for q in json.loads(f.read_text()):
                used.add(q["problem_id"])
    unused = [q for q in pool if q["problem_id"] not in used]
    rng = random.Random(SEED)
    return rng.sample(unused, min(N_FRESH, len(unused)))


def load_candidates():
    """Load 12 candidates with their full wisdom records."""
    val = json.loads((AUTO_DIR / "validation_log_parallel.json").read_text())
    cands = []
    for entry in val:
        for r in entry["results"]:
            cands.append({
                "tid": r["tid"],
                "aphorism": r["candidate"],
                "source": r.get("source", ""),
            })
    # enrich with signal/unpacked from success_distilled / cross_llm
    sd = cache_load(AUTO_DIR / "success_distilled_candidates.json", default=[])
    cl = cache_load(AUTO_DIR / "cross_llm_candidates.json", default=[])
    aph_to_rec = {r["aphorism"]: r for r in sd + cl if "aphorism" in r}
    for c in cands:
        rec = aph_to_rec.get(c["aphorism"], {})
        c["signal"] = rec.get("signal", "")
        c["unpacked_for_llm"] = rec.get("unpacked_for_llm", c["aphorism"])
    return cands


def solve(client, problem, wisdoms):
    try:
        r = client.generate(FRAME_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.2)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    if wisdoms:
        wb = "\n".join(f"• {w['aphorism']}: {w.get('unpacked_for_llm', '')[:200]}"
                        for w in wisdoms)
    else:
        wb = "(无)"
    try:
        r = client.generate(EXECUTE_PROMPT.format(
            frame=m.get("frame", "object_level"),
            critical_reframe=m.get("critical_reframe", ""),
            rewritten_problem=m.get("rewritten_problem", problem),
            wisdom_block=wb), max_tokens=900, temperature=0.2)
        return r["text"].strip()
    except Exception as e:
        return f"[turn1 err: {e}]"


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception:
        return "err"


def wilson(k, n):
    from scipy.stats import binomtest
    if n == 0: return (0.5, 0.5)
    return tuple(binomtest(k, n).proportion_ci(method="wilson"))


def main():
    print("=== Exp 47: Preregistered audit on fresh 30-pid split ===")
    print(f"  Frozen thresholds (from PREREGISTRATION_v2.md):")
    print(f"    inner-loop gate: wr_ext >= {INNER_THRESHOLD}")
    print(f"    L1 cross-family: wr_ext >= {L1_THRESHOLD}\n")

    fresh_pids = fresh_sample()
    print(f"  Fresh pids ({len(fresh_pids)}, seed={SEED}, disjoint from "
          f"sample_100/holdout_50/extend_50):")
    print(f"    domains: {sorted(set(p['problem_id'].rsplit('_', 1)[0] for p in fresh_pids))}")
    pid_to_prob = {p["problem_id"]: p["description"] for p in fresh_pids}

    cands = load_candidates()
    print(f"  Candidates: {len(cands)}\n")

    solver = cheap("gemini")
    haiku = cheap("claude_haiku")
    print(f"  Solver: {solver.model}; L1 judge: {haiku.model}\n")

    # Stage 1: generate base answers
    print(f"[1/4] Generating {len(fresh_pids)} base answers (no wisdom)...")
    base_answers = {}
    def gen_base(p):
        return p["problem_id"], solve(solver, p["description"], [])
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_base, p) for p in fresh_pids]
        for i, f in enumerate(as_completed(futs)):
            pid, ans = f.result()
            base_answers[pid] = ans
            if (i + 1) % 10 == 0:
                print(f"  base {i+1}/{len(fresh_pids)} ({time.time()-t0:.0f}s)")

    # Stage 2: generate ext answers per candidate
    print(f"\n[2/4] Generating {len(cands)} × {len(fresh_pids)} = "
          f"{len(cands)*len(fresh_pids)} ext answers...")
    ext_answers = {c["tid"]: {} for c in cands}
    def gen_ext(c, p):
        return c["tid"], p["problem_id"], solve(solver, p["description"], [c])
    tasks = [(c, p) for c in cands for p in fresh_pids]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_ext, c, p) for c, p in tasks]
        for f in as_completed(futs):
            tid, pid, ans = f.result()
            ext_answers[tid][pid] = ans
            done += 1
            if done % 30 == 0:
                print(f"  ext {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Stage 3: inner-loop judge (gemini)
    print(f"\n[3/4] Inner-loop judge (gemini) on {len(cands)} × {len(fresh_pids)} pairs...")
    inner_verdicts = {c["tid"]: {} for c in cands}
    def judge_inner(c, pid):
        b = base_answers.get(pid, ""); e = ext_answers[c["tid"]].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return c["tid"], pid, "missing"
        rng = random.Random((hash(pid) ^ hash(c["tid"])) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(solver, pid_to_prob[pid], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return c["tid"], pid, v

    inner_tasks = [(c, p["problem_id"]) for c in cands for p in fresh_pids]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_inner, c, pid) for c, pid in inner_tasks]
        for f in as_completed(futs):
            tid, pid, v = f.result()
            inner_verdicts[tid][pid] = v
            done += 1
            if done % 30 == 0:
                print(f"  inner judge {done}/{len(inner_tasks)} ({time.time()-t0:.0f}s)")

    # Aggregate inner-loop wr
    inner_summary = []
    for c in cands:
        v = inner_verdicts[c["tid"]]
        ne = sum(1 for x in v.values() if x == "ext")
        nb = sum(1 for x in v.values() if x == "base")
        nt = sum(1 for x in v.values() if x == "tie")
        n_eff = ne + nb
        wr = ne / n_eff if n_eff else 0.5
        inner_pass = wr >= INNER_THRESHOLD
        inner_summary.append({"tid": c["tid"], "aphorism": c["aphorism"],
                                "ext": ne, "base": nb, "tie": nt,
                                "n_eff": n_eff, "wr_inner": wr,
                                "inner_pass": inner_pass})

    print(f"\n=== Inner-loop verdict on fresh data (frozen threshold {INNER_THRESHOLD}) ===")
    print(f"{'tid':12s} {'wr_inner':>10s}  {'n_eff':>6s} ties  {'pass?':>6s}")
    n_inner_pass = 0
    for s in inner_summary:
        marker = "PASS" if s["inner_pass"] else "REVERT"
        print(f"  {s['tid']:12s} {s['wr_inner']:>10.2f}  {s['n_eff']:>6d}  {s['tie']:>4d}  {marker}")
        if s["inner_pass"]:
            n_inner_pass += 1
    print(f"\n  {n_inner_pass}/{len(cands)} candidates pass the inner-loop gate on fresh data")

    # Stage 4: L1 cross-family on candidates that passed inner (or all if none pass)
    candidates_for_l1 = [s for s in inner_summary if s["inner_pass"]]
    if not candidates_for_l1:
        # Run L1 on top-3 by wr_inner anyway, for comparison
        candidates_for_l1 = sorted(inner_summary, key=lambda s: -s["wr_inner"])[:3]
        print(f"\n  No candidates passed inner-loop on fresh data.")
        print(f"  Running L1 on top-3 by wr_inner anyway, for comparison.")

    print(f"\n[4/4] L1 cross-family (claude-haiku) on {len(candidates_for_l1)} candidates...")
    l1_verdicts = {s["tid"]: {} for s in candidates_for_l1}
    l1_tasks = [(s, p["problem_id"]) for s in candidates_for_l1 for p in fresh_pids]

    def judge_l1(s, pid):
        b = base_answers.get(pid, ""); e = ext_answers[s["tid"]].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return s["tid"], pid, "missing"
        rng = random.Random((hash(pid) ^ hash(s["tid"])) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(haiku, pid_to_prob[pid], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return s["tid"], pid, v

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_l1, s, pid) for s, pid in l1_tasks]
        for f in as_completed(futs):
            tid, pid, v = f.result()
            l1_verdicts[tid][pid] = v
            done += 1
            if done % 20 == 0:
                print(f"  l1 judge {done}/{len(l1_tasks)} ({time.time()-t0:.0f}s)")

    # Final aggregate
    print(f"\n=== Final preregistered verdict (frozen thresholds) ===")
    print(f"{'tid':12s} {'wr_inner':>9s} {'inner':>6s}  {'wr_L1':>8s} {'L1':>6s}  outcome")
    print("-" * 80)
    final_results = []
    for s in inner_summary:
        v_l1 = l1_verdicts.get(s["tid"], {})
        ne_l1 = sum(1 for x in v_l1.values() if x == "ext")
        nb_l1 = sum(1 for x in v_l1.values() if x == "base")
        n_eff_l1 = ne_l1 + nb_l1
        wr_l1 = ne_l1 / n_eff_l1 if n_eff_l1 else None
        l1_pass = (wr_l1 is not None) and (wr_l1 >= L1_THRESHOLD)

        if not s["inner_pass"]:
            outcome = "inner-FAIL"
        elif l1_pass:
            outcome = "BOTH PASS"  # would commit
        else:
            outcome = "inner-PASS, L1-FAIL"
        final_results.append({**s, "wr_l1": wr_l1, "n_eff_l1": n_eff_l1,
                                "l1_pass": l1_pass, "outcome": outcome})
        line = (f"  {s['tid']:12s} {s['wr_inner']:>9.2f} "
                f"{('PASS' if s['inner_pass'] else 'FAIL'):>6s}  ")
        if wr_l1 is not None:
            line += f"{wr_l1:>8.2f} {('PASS' if l1_pass else 'FAIL'):>6s}"
        else:
            line += " " * 16
        line += f"   {outcome}"
        print(line)

    n_both = sum(1 for r in final_results if r["outcome"] == "BOTH PASS")
    n_inner_only = sum(1 for r in final_results if r["outcome"] == "inner-PASS, L1-FAIL")
    n_inner_fail = sum(1 for r in final_results if r["outcome"] == "inner-FAIL")
    print(f"\n=== Headline (preregistered outcome categories) ===")
    print(f"  BOTH PASS (commit per PREREG): {n_both}/{len(cands)}")
    print(f"  inner-PASS, L1-FAIL (audit catches): {n_inner_only}/{len(cands)}")
    print(f"  inner-FAIL: {n_inner_fail}/{len(cands)}")
    print()
    print(f"  Original cycle: 3 KEEPs (W076, W077, W078) on sample_extend_50.")
    print(f"  Fresh cycle: {n_both} KEEP{'s' if n_both != 1 else ''} on fresh 30-pid split.")
    print()
    if n_inner_pass == 0:
        print("  Pre-specified outcome category 2 (Strong null replication): inner-loop")
        print("  produces ZERO accepts on fresh data. The original 3 KEEPs do not")
        print("  replicate to a held-out problem distribution.")
    elif n_inner_pass == n_inner_only:
        print("  Pre-specified outcome category 3 (Partial null): inner-loop accepts")
        print(f"  {n_inner_pass} candidates, but L1 audit catches all of them.")
    elif n_both > 0:
        print(f"  Pre-specified outcome category 1+ (Some replication): {n_both}")
        print(f"  candidate{'s' if n_both != 1 else ''} pass{'es' if n_both == 1 else ''} both inner-loop and L1 on fresh data.")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "fresh_pids": [p["problem_id"] for p in fresh_pids],
           "n_fresh": len(fresh_pids),
           "seed": SEED,
           "frozen_thresholds": {"inner": INNER_THRESHOLD, "L1": L1_THRESHOLD},
           "judges": {"inner": solver.model, "L1": haiku.model},
           "summary": final_results,
           "outcome_counts": {"BOTH_PASS": n_both, "inner_PASS_L1_FAIL": n_inner_only,
                                "inner_FAIL": n_inner_fail}}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
