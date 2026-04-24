"""Exp 26 — Solver-family ablation.

Directly addresses reviewer objection #2 from the harsh review:
  "Single solver family (gemini-3-flash) throughout; findings may be
   solver-specific."

Fix: re-run a reduced v20-like pipeline (frame+rewrite + answer) on
N=20 sample_extend_50 pids using 3 solver families (gemini, claude-
haiku, gpt-5.4-mini) — base library + W076 ext. Then gemini-judge
the (base, ext) pairs. Check if "adding W076 to library helps" holds
regardless of solver family.

Scope: just W076 to stay in time budget (can expand later).
20 pids × 3 families × 2 libs (base+ext) = 120 solver calls + 60 judges.

Minimal 2-turn v20-like solver:
  Turn 0: frame+rewrite JSON
  Turn 1: answer using frame + rewrite + retrieved wisdom

Uses same prompts as phase2_v20_framework.py but with per-family client.
"""

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase four"))

from model_router import cheap, cheap_panel
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp26_solver_ablation_log.json"
SAMPLE_FILE = "sample_extend_50.json"


FRAME_REWRITE_PROMPT = """对下面的问题，产生诊断 frame 和重写版本。

## 原题
{problem}

## 输出 JSON（不要代码块）
{{"frame": "hybrid 或 paradigm 或 object_level",
  "critical_reframe": "一句话点出问题真正要回答什么（30-80字）",
  "rewritten_problem": "重写版本，显式化 frame 下的关键维度（120-250字）"}}
"""

EXECUTE_PROMPT = """# 解决下面的问题。

## PRIMARY FRAME
- frame: {frame}
- critical reframe: {critical_reframe}

## 问题（重写后）
{rewritten_problem}

## 次要参考 wisdom
{wisdom_block}

## 要求
- 服从 PRIMARY FRAME
- 不超过 500 字

开始："""

JUDGE_PROMPT = """你是方法论评审专家。下面是同一个问题的两个解答。

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

## 评审
四个维度：问题理解、分析深度、结构化程度、实用性。

输出 JSON（不要代码块）：
{{"winner": "A"或"B"或"tie", "score_a": 1-10整数, "score_b": 1-10整数,
  "reasoning": "80字内说明胜因"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def solve_v20_lite(client, problem, wisdoms):
    """2-turn v20-like: frame+rewrite → execute. Returns answer text."""
    # Turn 0
    try:
        r = client.generate(FRAME_REWRITE_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.3)
        meta = parse_json_from_llm(r["text"])
        frame = meta.get("frame", "hybrid")
        crit = meta.get("critical_reframe", "")
        rewrite = meta.get("rewritten_problem", problem)
    except Exception as e:
        return f"[turn0 err: {e}]"
    # Turn 1
    wb = "\n\n".join(f"• {w['aphorism']}: {w.get('unpacked_for_llm', '')[:200]}"
                       for w in wisdoms) if wisdoms else "(无)"
    try:
        r = client.generate(EXECUTE_PROMPT.format(
            frame=frame, critical_reframe=crit,
            rewritten_problem=rewrite, wisdom_block=wb),
            max_tokens=1000, temperature=0.3)
        return r["text"].strip()
    except Exception as e:
        return f"[turn1 err: {e}]"


def judge_pair(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception:
        return "err"


def load_w076():
    cands = cache_load(AUTO_DIR / "success_distilled_candidates.json", default=[])
    for c in cands:
        if c.get("aphorism") == "凡益之道，与时偕行":
            return c
    raise ValueError("W076 candidate not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    problems = cache_load(CACHE / SAMPLE_FILE)
    problems = [p for p in problems if "description" in p][: args.n]
    print(f"Testing on {len(problems)} problems from {SAMPLE_FILE}")

    w076 = load_w076()
    ext_wisdoms = [w076]   # ext library adds just W076 for this ablation
    base_wisdoms = []       # base = no extra wisdom (cleanest comparison)

    # 3 cheap solver families
    solvers = cheap_panel()
    print(f"Solver panel: {[c.model for c in solvers]}\n")

    # Judge — use gemini (same as main study)
    judge = cheap("gemini")

    results = {c.family: {"base": {}, "ext": {}, "judge_wr": None} for c in solvers}

    # Generate answers: for each (solver, lib, pid) — 3 × 2 × N = 6N calls
    print(f"[1/2] Generating {len(solvers)*2*len(problems)} answers...")
    tasks = []
    for s in solvers:
        for lib_name, wisdoms in (("base", base_wisdoms), ("ext", ext_wisdoms)):
            for p in problems:
                tasks.append((s, lib_name, p, wisdoms))

    def run_one(task):
        s, lib_name, p, w = task
        return s.family, lib_name, p["problem_id"], solve_v20_lite(s, p["description"], w)

    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=6) as ex:
        futs = [ex.submit(run_one, t) for t in tasks]
        for f in as_completed(futs):
            try:
                fam, lib, pid, ans = f.result()
                results[fam][lib][pid] = ans
                done += 1
                if done % 20 == 0:
                    dt = time.time() - t0
                    eta = dt/done*(len(tasks)-done)
                    print(f"  answers {done}/{len(tasks)} ({dt:.0f}s, eta {eta:.0f}s)")
            except Exception as e:
                print(f"  [err] {e}")

    print(f"\n[2/2] Judging ext-vs-base per family (gemini judge)...")
    for fam in results:
        verdicts = []
        for p in problems:
            pid = p["problem_id"]
            b = results[fam]["base"].get(pid, "")
            e = results[fam]["ext"].get(pid, "")
            if not b or not e or "err" in b[:20] or "err" in e[:20]:
                continue
            rng = random.Random(hash(pid) % (2**32))
            if rng.random() < 0.5:
                left, right, ext_was = e, b, "A"
            else:
                left, right, ext_was = b, e, "B"
            w = judge_pair(judge, p["description"], left, right)
            if w == "tie": verdicts.append("tie")
            elif w in ("A", "B"):
                verdicts.append("ext" if w == ext_was else "base")
        n_ext = sum(1 for v in verdicts if v == "ext")
        n_base = sum(1 for v in verdicts if v == "base")
        n_tie = sum(1 for v in verdicts if v == "tie")
        tot = n_ext + n_base
        wr = n_ext / tot if tot else 0.5
        results[fam]["judge_wr"] = {"ext_wins": n_ext, "base_wins": n_base,
                                     "ties": n_tie, "wr_ext": wr,
                                     "n_decided": tot}
        print(f"  {fam:15s} ext={n_ext} base={n_base} ties={n_tie}  wr_ext={wr:.2f}")

    print(f"\n=== SUMMARY ===")
    print(f"Question: does W076 help regardless of solver family?")
    wrs = [results[f]["judge_wr"]["wr_ext"] for f in results
            if results[f]["judge_wr"]]
    print(f"  per-family wr_ext: {wrs}")
    if len(wrs) >= 2:
        mean_wr = sum(wrs) / len(wrs)
        all_above = all(w > 0.5 for w in wrs)
        print(f"  mean wr_ext: {mean_wr:.2f}")
        print(f"  all families >= 0.5? {all_above}")
        print(f"  conclusion: {'W076 replicates across solver families' if all_above and mean_wr>=0.55 else 'W076 is solver-family-specific OR weak'}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "target_wisdom": "W076", "n_problems": len(problems),
           "solver_panel": [s.model for s in solvers],
           "judge": judge.model,
           "results": results}
    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(out)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
