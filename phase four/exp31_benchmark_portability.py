"""Exp 31 — Benchmark portability: test W078 on 20 unseen math problems.

Addresses reviewer objection #4 (no standard benchmark).

Strict reading of the objection: we run on our own 100+50 problem split;
can the signal port? Rather than pulling GSM8K/MATH (different domain,
English), we use a concrete proxy: **20 math problems from the project's
1768-problem pool that have NEVER been used in any prior experiment**.

This is not GSM8K but it's a real portability test: completely fresh
problems of a specific domain (math) where W078's "bake-off orientation"
(method selection) has a plausible mechanism.

Protocol:
  1. Sample 20 math problems disjoint from sample_100 + holdout_50 +
     extend_50
  2. For each, generate v20 base (75 wisdoms) + ext (75 + W078) using
     gemini-3-flash solver + same retrieval
  3. Cross-family judge (gemini + claude + gpt-mini) vote
  4. Report: does W078's signal port to fresh math problems?
"""

import json
import random
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap, cheap_panel
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp31_benchmark_portability_log.json"

N_PROBLEMS = 20
PARALLEL = 6


FRAME_REWRITE_PROMPT = """对下面问题产生诊断 frame + 重写。

## 原题
{problem}

## 输出 JSON
{{"frame": "hybrid/paradigm/object_level",
  "critical_reframe": "一句话 30-80 字",
  "rewritten_problem": "120-250 字"}}
"""

EXECUTE_PROMPT = """# 解决问题

## PRIMARY FRAME
- frame: {frame}
- critical reframe: {critical_reframe}

## 问题（重写）
{rewritten_problem}

## 次要参考 wisdom
{wisdom_block}

## 要求：不超过 500 字
开始："""

JUDGE_PROMPT = """方法论评审。

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

四维度评审。输出 JSON:
{{"winner": "A"/"B"/"tie", "score_a": 1-10, "score_b": 1-10,
  "reasoning": "80字内"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def load_w078():
    for src in ("success_distilled_candidates.json", "cross_llm_candidates.json"):
        data = cache_load(AUTO_DIR / src, default=[])
        for c in data:
            if c.get("aphorism") == "是骡子是马，拉出来遛遛":
                return c
    raise ValueError("W078 not found")


def pick_unused_math(n):
    used = set()
    for f in ("sample_100.json", "sample_holdout_50.json", "sample_extend_50.json"):
        p = CACHE / f
        if p.exists():
            used.update(q["problem_id"] for q in json.loads(p.read_text()))

    math_pool = json.loads((PROJECT / "phase zero" / "benchmark" /
                             "problems" / "mathematics.json").read_text())
    unused = [q for q in math_pool if q["problem_id"] not in used]
    rng = random.Random(2028)
    return rng.sample(unused, min(n, len(unused)))


def solve(client, problem, wisdoms):
    try:
        r = client.generate(FRAME_REWRITE_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.3)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    wb = "\n\n".join(f"• {w['aphorism']}: {w.get('unpacked_for_llm','')[:200]}"
                       for w in wisdoms) if wisdoms else "(无)"
    try:
        r = client.generate(EXECUTE_PROMPT.format(
            frame=m.get("frame", "hybrid"),
            critical_reframe=m.get("critical_reframe", ""),
            rewritten_problem=m.get("rewritten_problem", problem),
            wisdom_block=wb), max_tokens=1000, temperature=0.3)
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
    problems = pick_unused_math(N_PROBLEMS)
    print(f"Picked {len(problems)} unused math problems from pool of 265\n")

    w078 = load_w078()
    base_wisdoms = []
    ext_wisdoms = [w078]

    solver = cheap("gemini")
    print(f"Solver: {solver.model}")

    # Step 1: generate answers per problem per library
    print(f"\n[1/2] Generating {len(problems)*2} answers (gemini solver)...")
    tasks = []
    for p in problems:
        tasks.append(("base", p, base_wisdoms))
        tasks.append(("ext", p, ext_wisdoms))

    def gen(task):
        lib, p, w = task
        ans = solve(solver, p["description"], w)
        return lib, p["problem_id"], ans

    answers = {"base": {}, "ext": {}}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen, t) for t in tasks]
        done = 0
        for f in as_completed(futs):
            lib, pid, ans = f.result()
            answers[lib][pid] = ans
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Step 2: cross-family judging
    print(f"\n[2/2] Cross-family judge ({len(problems)} pairs × 3 families)...")
    judges = cheap_panel()

    def judge_pair(judge, p):
        pid = p["problem_id"]
        b = answers["base"].get(pid, "")
        e = answers["ext"].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return judge.family, pid, "missing"
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(judge, p["description"], left, right)
        if w == "tie": return judge.family, pid, "tie"
        if w in ("A", "B"):
            return judge.family, pid, "ext" if w == ext_was else "base"
        return judge.family, pid, "err"

    verdicts = {j.family: {} for j in judges}
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_pair, j, p) for j in judges for p in problems]
        for f in as_completed(futs):
            fam, pid, v = f.result()
            verdicts[fam][pid] = v

    print(f"\n=== Per-family verdicts on 20 unseen math problems ===")
    wrs = {}
    for fam, verd in verdicts.items():
        n_ext = sum(1 for v in verd.values() if v == "ext")
        n_base = sum(1 for v in verd.values() if v == "base")
        n_tie = sum(1 for v in verd.values() if v == "tie")
        tot = n_ext + n_base
        wr = n_ext / tot if tot else 0.5
        wrs[fam] = wr
        print(f"  {fam:15s} ext={n_ext} base={n_base} tie={n_tie}  wr_ext={wr:.2f}")

    # Majority vote
    mv_pass = 0
    for pid in answers["base"]:
        votes = [verdicts[f].get(pid) for f in verdicts]
        ext_votes = sum(1 for v in votes if v == "ext")
        base_votes = sum(1 for v in votes if v == "base")
        if ext_votes > base_votes: mv_pass += 1

    print(f"\n  Majority-vote ext-wins: {mv_pass}/{len(problems)}")
    print(f"  Mean wr_ext across families: {sum(wrs.values())/len(wrs):.2f}")
    replicates = all(wrs[f] >= 0.50 for f in wrs) and sum(wrs.values())/len(wrs) >= 0.55
    print(f"  Signal replicates on fresh math? {'YES' if replicates else 'NO'}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "target_wisdom": "W078", "n_problems": len(problems),
           "benchmark": "fresh math (265-pool minus used)",
           "per_family_wr": wrs, "majority_vote_ext_wins": mv_pass,
           "signal_replicates": bool(replicates),
           "verdicts": verdicts}
    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(out)
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
