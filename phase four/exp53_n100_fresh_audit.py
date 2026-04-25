"""Exp 53 — Scale Exp 47 to n=100 fresh pids (R10/R11/R14 Q1).

The R8–R16 reviewers consistently flagged Exp 47 (n=30) as
underpowered: at the original 0.60 L1 threshold only W078 passed,
and joint posteriors were 0.37–0.54. Reviewer Q1: ``If Exp 47 were
rerun at n=100, would the headline survive?''

This experiment answers that directly. Same protocol as Exp 47:
- 12 original candidates evaluated unchanged
- Inner gate: gemini-3-flash, threshold 0.60 (preregistered)
- L1 cross-family: claude-haiku-4.5, threshold 0.55 (preregistered)
- Both thresholds frozen at the time of PREREGISTRATION_v2.md
  (commit 58c7b55 on 2026-04-25 08:03:45 +0800)
- Fresh pids disjoint from sample_100/holdout_50/extend_50
  *and* from the 30 pids used in Exp 47

This is NOT a fresh full-loop (candidate generation is not rerun;
the original 12 candidates are held fixed). It is a scaled-up
preregistered fresh-data replication that addresses the n=30
underpowering directly.

Cost: 100 pids × 12 candidates × (1 gen base + 1 gen ext + 1 gemini
judge) + 100 × 3 inner-passers × 1 L1 judge ≈ 2700 cheap-tier
API calls. ~$25-40.
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
OUT_LOG = AUTO_DIR / "exp53_n100_fresh_audit_log.json"

# Frozen thresholds from PREREGISTRATION_v2.md (line 27)
INNER_THRESHOLD = 0.60
L1_THRESHOLD = 0.55
SEED = 2027  # different from Exp 47's seed=2026 to ensure disjoint pids
N_FRESH = 100
PARALLEL = 6


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
    pool = []
    for f in (PROJECT / "phase zero" / "benchmark" / "problems").glob("*.json"):
        for q in json.loads(f.read_text()):
            if q.get("description"):
                pool.append(q)
    return pool


def fresh_sample():
    """Pick 100 pids disjoint from sample_100/holdout_50/extend_50 AND from
    Exp 47's 30 fresh pids."""
    pool = load_full_pool()
    used = set()
    for fn in ["sample_100.json", "sample_holdout_50.json",
                "sample_extend_50.json"]:
        f = CACHE / fn
        if f.exists():
            for q in json.loads(f.read_text()):
                used.add(q["problem_id"])
    # Exclude Exp 47's 30 fresh pids
    e47 = cache_load(AUTO_DIR / "exp47_preregistered_fresh_loop_log.json")
    if e47 and "fresh_pids" in e47:
        used.update(e47["fresh_pids"])
    unused = [q for q in pool if q["problem_id"] not in used]
    rng = random.Random(SEED)
    return rng.sample(unused, min(N_FRESH, len(unused)))


def load_candidates():
    val = json.loads((AUTO_DIR / "validation_log_parallel.json").read_text())
    cands = []
    for entry in val:
        for r in entry["results"]:
            cands.append({
                "tid": r["tid"],
                "aphorism": r["candidate"],
                "source": r.get("source", ""),
            })
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
        r = client.generate(JUDGE_PROMPT.format(problem=problem,
                                                 answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception:
        return "err"


def main():
    print(f"=== Exp 53: Fresh n={N_FRESH} audit (preregistered protocol) ===")
    print(f"  Frozen thresholds:")
    print(f"    inner-loop gate: wr_ext >= {INNER_THRESHOLD}")
    print(f"    L1 cross-family: wr_ext >= {L1_THRESHOLD}\n")

    fresh_pids = fresh_sample()
    print(f"  Fresh pids ({len(fresh_pids)}, seed={SEED}, "
          f"disjoint from sample_100/holdout_50/extend_50 AND from "
          f"Exp 47's 30 pids)")
    pid_to_prob = {p["problem_id"]: p["description"] for p in fresh_pids}

    cands = load_candidates()
    print(f"  Candidates: {len(cands)}\n")

    solver = cheap("gemini")
    haiku = cheap("claude_haiku")
    print(f"  Solver: {solver.model}; L1 judge: {haiku.model}\n")

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
            if (i + 1) % 20 == 0:
                print(f"  base {i+1}/{len(fresh_pids)} ({time.time()-t0:.0f}s)")

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
            if done % 60 == 0:
                print(f"  ext {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    print(f"\n[3/4] Inner-loop judge (gemini) on {len(cands)} × "
          f"{len(fresh_pids)} pairs...")
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
            if done % 60 == 0:
                print(f"  inner judge {done}/{len(inner_tasks)} ({time.time()-t0:.0f}s)")

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

    print(f"\n=== Inner-loop verdict on n={N_FRESH} fresh data ===")
    print(f"{'tid':12s} {'wr_inner':>10s}  {'n_eff':>6s} {'ties':>4s}  {'pass':>6s}")
    n_inner_pass = 0
    for s in inner_summary:
        marker = "PASS" if s["inner_pass"] else "REVERT"
        print(f"  {s['tid']:12s} {s['wr_inner']:>10.3f}  "
              f"{s['n_eff']:>6d} {s['tie']:>4d}  {marker}")
        if s["inner_pass"]:
            n_inner_pass += 1
    print(f"\n  {n_inner_pass}/{len(cands)} candidates pass inner gate at "
          f"{INNER_THRESHOLD}")

    # Stage 4: L1 on inner-passers (and top-3 by wr if no passers)
    candidates_for_l1 = [s for s in inner_summary if s["inner_pass"]]
    if not candidates_for_l1:
        candidates_for_l1 = sorted(inner_summary, key=lambda s: -s["wr_inner"])[:3]
        print(f"\n  No candidates passed inner gate; running L1 on top-3 anyway.")

    print(f"\n[4/4] L1 cross-family (claude-haiku) on {len(candidates_for_l1)} "
          f"candidates × {len(fresh_pids)} pids...")
    l1_verdicts = {s["tid"]: {} for s in candidates_for_l1}
    l1_tasks = [(s, p["problem_id"]) for s in candidates_for_l1
                  for p in fresh_pids]

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
            if done % 30 == 0:
                print(f"  l1 judge {done}/{len(l1_tasks)} ({time.time()-t0:.0f}s)")

    print(f"\n=== Final n={N_FRESH} verdict (frozen thresholds) ===")
    print(f"{'tid':12s} {'wr_inner':>9s} {'inner':>6s}  {'wr_L1':>8s} "
          f"{'L1@.55':>7s} {'L1@.60':>7s}  outcome")
    print("-" * 90)
    final_results = []
    for s in inner_summary:
        v_l1 = l1_verdicts.get(s["tid"], {})
        ne_l1 = sum(1 for x in v_l1.values() if x == "ext")
        nb_l1 = sum(1 for x in v_l1.values() if x == "base")
        n_eff_l1 = ne_l1 + nb_l1
        wr_l1 = ne_l1 / n_eff_l1 if n_eff_l1 else None
        l1_pass_55 = (wr_l1 is not None) and (wr_l1 >= 0.55)
        l1_pass_60 = (wr_l1 is not None) and (wr_l1 >= 0.60)

        if not s["inner_pass"]:
            outcome_55 = "inner-FAIL"
            outcome_60 = "inner-FAIL"
        elif l1_pass_55:
            outcome_55 = "BOTH PASS"
        else:
            outcome_55 = "L1-FAIL"
        if s["inner_pass"]:
            outcome_60 = "BOTH PASS@.60" if l1_pass_60 else "L1-FAIL@.60"

        wr_l1_s = f"{wr_l1:.3f}" if wr_l1 is not None else "---"
        i_mark = "PASS" if s["inner_pass"] else "FAIL"
        l55 = "PASS" if l1_pass_55 else ("---" if wr_l1 is None else "FAIL")
        l60 = "PASS" if l1_pass_60 else ("---" if wr_l1 is None else "FAIL")
        print(f"  {s['tid']:12s} {s['wr_inner']:>9.3f} {i_mark:>6s}  "
              f"{wr_l1_s:>8s} {l55:>7s} {l60:>7s}  {outcome_55}")

        s["wr_l1"] = wr_l1
        s["n_eff_l1"] = n_eff_l1
        s["ext_l1"] = ne_l1
        s["base_l1"] = nb_l1
        s["l1_pass_55"] = l1_pass_55
        s["l1_pass_60"] = l1_pass_60
        s["outcome_pre"] = outcome_55
        s["outcome_strict"] = outcome_60
        final_results.append(s)

    # Summary
    n_pre = sum(1 for s in final_results if s["inner_pass"] and s.get("l1_pass_55"))
    n_strict = sum(1 for s in final_results if s["inner_pass"] and s.get("l1_pass_60"))
    print(f"\n=== Headline ===")
    print(f"  At preregistered policy (inner>=0.60 AND L1>=0.55): "
          f"{n_pre}/{len(cands)} pass")
    print(f"  At strict policy       (inner>=0.60 AND L1>=0.60): "
          f"{n_strict}/{len(cands)} pass")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "fresh_pids": [p["problem_id"] for p in fresh_pids],
           "n_fresh": len(fresh_pids), "seed": SEED,
           "frozen_thresholds": {"inner": INNER_THRESHOLD,
                                  "L1_pre": 0.55, "L1_strict": 0.60},
           "judges": [solver.model, haiku.model],
           "summary": final_results,
           "n_pass_pre": n_pre, "n_pass_strict": n_strict,
           "inner_verdicts": inner_verdicts,
           "l1_verdicts": l1_verdicts}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
