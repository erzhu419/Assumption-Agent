"""Experiment 1: Cross-judge validation.

Re-judge the SAME (base, ext) answer pairs that produced W076/W077/W078
with a different LLM family (Claude Opus 4.6). If the decision flips,
the original gate was judge-dependent; if it holds, self-adjudication
is cross-judge stable.

Inputs:
  - cached base answers: _valp_v20p1_base_answers.json  (for W076, W077)
  - cached base answers: _valp_v20_base_answers.json    (for W078, post-prune)
  - cached ext  answers: _valp_v20_ext_WCAND05_answers.json   (W076)
                         _valp_v20_ext_WCAND10_answers.json   (W077)
                         _valp_v20_ext_WCROSSL01_answers.json (W078)
  - held-out 50 problems

Output:
  phase four/autonomous/exp1_cross_judge_log.json
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))

from claude_proxy_client import ClaudeProxyClient
from llm_client import parse_json_from_llm


CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp1_cross_judge_log.json"

HOLDOUT_SAMPLE = "sample_holdout_50.json"
PARALLEL_JUDGES = 6

# The 3 KEEP decisions to revalidate
KEEP_CASES = [
    {
        "wid": "W076", "aphorism": "凡益之道，与时偕行",
        "base_answers": "_valp_v20p1_base_answers.json",
        "ext_answers":  "_valp_v20_ext_WCAND05_answers.json",
        "original_wr_ext": 0.64,
        "original_judge": "gemini-3-flash",
    },
    {
        "wid": "W077", "aphorism": "没有调查，就没有发言权",
        "base_answers": "_valp_v20p1_base_answers.json",
        "ext_answers":  "_valp_v20_ext_WCAND10_answers.json",
        "original_wr_ext": 0.60,
        "original_judge": "gemini-3-flash",
    },
    {
        "wid": "W078", "aphorism": "是骡子是马，拉出来遛遛",
        "base_answers": "_valp_v20_base_answers.json",  # v20.3 base
        "ext_answers":  "_valp_v20_ext_WCROSSL01_answers.json",
        "original_wr_ext": 0.60,
        "original_judge": "gemini-3-flash",
    },
]


# Same judge prompt wording as cached_framework.JUDGE_PROMPT,
# run via Claude Opus 4.6.
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


def judge_one(client, problem, a, b):
    prompt = JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b)
    try:
        r = client.generate(prompt, max_tokens=400, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie"), v.get("score_a", 0), v.get("score_b", 0), \
               v.get("reasoning", "")
    except Exception as e:
        return "error", 0, 0, f"err: {e}"


def judge_ab_parallel(problems, ans_base, ans_ext, judge_client):
    """For each problem: judge ext vs base. Randomize side per PID seed."""
    def task(p):
        pid = p["problem_id"]
        ba, ea = ans_base.get(pid), ans_ext.get(pid)
        if not ba or not ea:
            return pid, "missing", 0, 0, ""
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = ea, ba, "A"
        else:
            left, right, ext_was = ba, ea, "B"
        winner, sa, sb, rs = judge_one(judge_client, p["description"], left, right)
        # Normalize: is ext winner?
        if winner == "tie":
            return pid, "tie", sa, sb, rs
        if (winner == ext_was):
            return pid, "ext", sa, sb, rs
        return pid, "base", sa, sb, rs

    out = {}
    with ThreadPoolExecutor(max_workers=PARALLEL_JUDGES) as ex:
        futures = {ex.submit(task, p): p["problem_id"] for p in problems}
        for f in as_completed(futures):
            try:
                pid, who, sa, sb, rs = f.result()
                out[pid] = {"winner": who, "score_a": sa, "score_b": sb, "reasoning": rs}
            except Exception as e:
                print(f"    [err] {e}")
    return out


def summarize(verdicts):
    c = {"ext": 0, "base": 0, "tie": 0, "missing": 0, "error": 0}
    for v in verdicts.values():
        c[v.get("winner", "error")] = c.get(v.get("winner", "error"), 0) + 1
    total_decided = c["ext"] + c["base"]
    wr_ext = c["ext"] / total_decided if total_decided else 0.5
    return {"n_ext": c["ext"], "n_base": c["base"], "n_tie": c["tie"],
            "n_missing": c["missing"], "n_error": c["error"],
            "total_decided": total_decided, "wr_ext": wr_ext}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-threshold", type=float, default=0.60)
    args = ap.parse_args()

    problems = json.loads((CACHE / HOLDOUT_SAMPLE).read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]
    print(f"Held-out problems: {len(problems)}")

    judge = ClaudeProxyClient()
    print(f"Judge: {judge.model}  via {judge.provider}\n")

    results = []
    t0 = time.time()
    for case in KEEP_CASES:
        print(f"=== [{case['wid']}] {case['aphorism']} ===")
        base_path = CACHE / "answers" / case["base_answers"]
        ext_path = CACHE / "answers" / case["ext_answers"]
        if not base_path.exists():
            print(f"  [SKIP] missing base answers: {base_path.name}")
            continue
        if not ext_path.exists():
            print(f"  [SKIP] missing ext answers: {ext_path.name}")
            continue
        ans_base = json.loads(base_path.read_text(encoding="utf-8"))
        ans_ext = json.loads(ext_path.read_text(encoding="utf-8"))
        print(f"  base: {len(ans_base)}, ext: {len(ans_ext)}")

        t1 = time.time()
        verdicts = judge_ab_parallel(problems, ans_base, ans_ext, judge)
        summary = summarize(verdicts)
        wr = summary["wr_ext"]
        claude_decision = "KEEP" if wr >= args.keep_threshold else "REVERT"
        agreement = "STABLE" if claude_decision == "KEEP" else "FLIP"
        print(f"  Claude judge: ext={summary['n_ext']} base={summary['n_base']} "
              f"ties={summary['n_tie']} missing={summary['n_missing']}  wr_ext={wr:.2f}  "
              f"({time.time()-t1:.0f}s)")
        print(f"  Original ({case['original_judge']}): wr={case['original_wr_ext']} → KEEP")
        print(f"  Claude (claude-opus-4-6):         wr={wr:.2f} → {claude_decision}  [{agreement}]\n")

        results.append({
            "wid": case["wid"],
            "aphorism": case["aphorism"],
            "original_judge": case["original_judge"],
            "original_wr_ext": case["original_wr_ext"],
            "cross_judge": judge.model,
            "cross_wr_ext": wr,
            "claude_decision": claude_decision,
            "agreement": agreement,
            "summary": summary,
            "verdicts": verdicts,
        })

    # Save log
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "judge_model": judge.model,
        "judge_provider": judge.provider,
        "n_problems": len(problems),
        "keep_threshold": args.keep_threshold,
        "wall_time_sec": int(time.time() - t0),
        "results": results,
    }
    log = json.loads(OUT_LOG.read_text()) if OUT_LOG.exists() else []
    log.append(entry)
    OUT_LOG.write_text(json.dumps(log, ensure_ascii=False, indent=2))
    print(f"Log → {OUT_LOG.name}")

    # Final summary
    print("\n=== FINAL ===")
    stable = sum(1 for r in results if r["agreement"] == "STABLE")
    print(f"  Cross-judge STABLE: {stable}/{len(results)}")
    for r in results:
        mark = "✓" if r["agreement"] == "STABLE" else "✗"
        print(f"   {mark} {r['wid']} ({r['aphorism']:15s}): "
              f"gemini={r['original_wr_ext']}  claude={r['cross_wr_ext']:.2f}")


if __name__ == "__main__":
    main()
