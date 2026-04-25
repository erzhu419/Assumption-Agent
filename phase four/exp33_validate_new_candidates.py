"""Exp 33 — Validate 9 new candidates from Exp 29 (scaled N).

Approach: cheap minimal A/B on 20 pids from sample_extend_50.
For each new candidate:
  - base answer: reuse cached _exp10_v20_base_answers if pids overlap,
    else generate with gemini solver + empty wisdom injection
  - ext answer: generate with gemini solver + [new candidate]
  - gemini-judge ext vs base

This gives per-candidate wr_ext on 20 pids. Combined with existing
12 candidates, total N = 22 with at least a minimal validation signal.
"""

import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp33_new_candidates_validation_log.json"

N_PIDS = 20
PARALLEL = 6


FRAME_PROMPT = """对下面问题产生 frame + 重写。

## 原题
{problem}

## 输出 JSON
{{"frame": "hybrid/paradigm/object_level",
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

JUDGE_PROMPT = """方法论评审。

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

输出 JSON:
{{"winner": "A"/"B"/"tie", "score_a": 1-10, "score_b": 1-10,
  "reasoning": "80字内"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def solve(client, problem, wisdoms):
    try:
        r = client.generate(FRAME_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.3)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    wb = "\n".join(f"• {w['aphorism']}: {w.get('unpacked_for_llm','')[:180]}"
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


def judge(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem,
                                                  answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception:
        return "err"


def main():
    extra = cache_load(AUTO_DIR / "exp29_extra_candidates.json", default=[])
    # dedup by aphorism to be safe (known: 不谋全局者 overlap with WCAND09)
    seen = set(); uniq = []
    for c in extra:
        a = c["aphorism"]
        if a in seen: continue
        seen.add(a); uniq.append(c)
    print(f"Extra candidates from Exp 29: {len(extra)} total, {len(uniq)} unique")

    problems = json.loads((CACHE / "sample_extend_50.json").read_text())
    problems = [p for p in problems if "description" in p][: N_PIDS]
    print(f"Using {len(problems)} pids from sample_extend_50\n")

    solver = cheap("gemini")
    judge_client = cheap("gemini")

    # Step 1: one base answer per pid (no extra wisdom)
    print(f"[1/3] Generating {len(problems)} base answers...")
    base_answers = {}
    def gen_base(p):
        return p["problem_id"], solve(solver, p["description"], [])
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen_base, p) for p in problems]
        for f in as_completed(futs):
            pid, ans = f.result()
            base_answers[pid] = ans

    # Step 2: per-candidate ext answers + judge
    results = []
    print(f"\n[2/3] For each of {len(uniq)} candidates: ext + judge on {len(problems)} pids...")
    for i, cand in enumerate(uniq):
        print(f"\n  [{i+1}/{len(uniq)}] {cand['aphorism']}")
        # Gen ext
        ext_answers = {}
        def gen_ext(p):
            return p["problem_id"], solve(solver, p["description"], [cand])
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(gen_ext, p) for p in problems]
            for f in as_completed(futs):
                pid, ans = f.result()
                ext_answers[pid] = ans
        # Judge
        n_ext = n_base = n_tie = 0
        for p in problems:
            pid = p["problem_id"]
            b = base_answers.get(pid, "")
            e = ext_answers.get(pid, "")
            if not b or not e or b.startswith("[") or e.startswith("["):
                continue
            rng = random.Random(hash(pid) % (2**32))
            if rng.random() < 0.5:
                left, right, ext_was = e, b, "A"
            else:
                left, right, ext_was = b, e, "B"
            w = judge(judge_client, p["description"], left, right)
            if w == "tie": n_tie += 1
            elif w == ext_was: n_ext += 1
            elif w in ("A", "B"): n_base += 1
        tot = n_ext + n_base
        wr = n_ext / tot if tot else 0.5
        print(f"      ext={n_ext} base={n_base} tie={n_tie}  wr_ext={wr:.2f}  "
              f"({time.time()-t0:.0f}s)")
        results.append({
            "aphorism": cand["aphorism"],
            "source": cand.get("source", ""),
            "cluster_size": cand.get("_cluster_size", "?"),
            "novelty_sim": cand.get("novelty_sim", 0),
            "wr_ext": wr, "ext_wins": n_ext,
            "base_wins": n_base, "ties": n_tie, "n": tot,
        })

    # Step 3: summary
    print(f"\n[3/3] Summary — new candidates' wr_ext on extend_50:")
    print(f"{'aphorism':30s} {'cluster_n':10s} {'wr_ext':8s} {'verdict'}")
    print("-" * 70)
    n_pass_at_0_60 = 0
    for r in results:
        v = "PASS" if r["wr_ext"] >= 0.60 else "REVERT"
        if v == "PASS": n_pass_at_0_60 += 1
        print(f"  {r['aphorism'][:28]:30s} {str(r['cluster_size']):10s} "
              f"{r['wr_ext']:<8.2f} {v}")
    print(f"\n  PASS at +10pp threshold (wr >= 0.60): {n_pass_at_0_60}/{len(results)}")
    print(f"  Mean wr_ext: {sum(r['wr_ext'] for r in results)/len(results):.2f}")

    # Combined with original inner-loop candidates
    orig_log = cache_load(AUTO_DIR / "validation_log_parallel.json", default=[])
    orig_pass = 0
    orig_total = 0
    for entry in orig_log:
        for r in entry.get("results", []):
            orig_total += 1
            if r.get("decision") == "KEEP": orig_pass += 1
    combined_total = orig_total + len(uniq)
    print(f"\n  Combined: {orig_pass} (from original {orig_total}) + "
          f"{n_pass_at_0_60} (from {len(uniq)} new) "
          f"= {orig_pass + n_pass_at_0_60} / {combined_total}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "n_pids": len(problems),
           "n_new_candidates": len(uniq),
           "solver": solver.model,
           "judge": judge_client.model,
           "results": results,
           "n_pass_at_0.60": n_pass_at_0_60,
           "combined_n_total": combined_total,
           "combined_n_pass": orig_pass + n_pass_at_0_60}
    prev = cache_load(OUT_LOG, default=[]) or []
    prev.append(out)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
