"""Exp 11: Third-family judge validation.

Gemini gave W076/W077/W078 KEEP (wr = 0.64, 0.60, 0.60).
Claude Opus flipped all three (wr = 0.40, 0.47, 0.51).

Question: is this a gemini-is-loose problem or a claude-is-strict problem?
A third family (GPT-5.4) tells us which:
  - If GPT-5.4 wr agrees with Claude (< 0.55) → gemini is loose
  - If GPT-5.4 wr agrees with gemini (>= 0.60) → claude is strict
  - If GPT-5.4 is in between → there is no unanimous truth; all families
    have systematic biases.

Uses the same 50 held-out answer pairs and same per-PID side randomization.
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

from gpt5_client import GPT5Client
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp11_third_family_log.json"
HOLDOUT_SAMPLE = "sample_holdout_50.json"
PARALLEL = 6

KEEPS = [
    # Three original KEEPs — Claude said REVERT on all
    {"wid": "W076", "cid": "WCAND05", "aphorism": "凡益之道，与时偕行",
     "base": "_valp_v20p1_base_answers.json",
     "ext": "_valp_v20_ext_WCAND05_answers.json",
     "gemini_wr": 0.64, "claude_wr": 0.40, "case_type": "gemini_high_claude_low"},
    {"wid": "W077", "cid": "WCAND10", "aphorism": "没有调查，就没有发言权",
     "base": "_valp_v20p1_base_answers.json",
     "ext": "_valp_v20_ext_WCAND10_answers.json",
     "gemini_wr": 0.60, "claude_wr": 0.47, "case_type": "gemini_high_claude_low"},
    {"wid": "W078", "cid": "WCROSSL01", "aphorism": "是骡子是马，拉出来遛遛",
     "base": "_valp_v20_base_answers.json",
     "ext": "_valp_v20_ext_WCROSSL01_answers.json",
     "gemini_wr": 0.60, "claude_wr": 0.51, "case_type": "gemini_high_claude_low"},
    # Two REVERT candidates Exp 9 revealed as "Claude says KEEP" — inverted disagreement
    {"wid": None, "cid": "WCAND07", "aphorism": "亲兄弟，明算账",
     "base": "_valp_v20p1_base_answers.json",
     "ext": "_valp_v20_ext_WCAND07_answers.json",
     "gemini_wr": 0.56, "claude_wr": 0.62, "case_type": "gemini_low_claude_high"},
    {"wid": None, "cid": "WCAND11", "aphorism": "若不是品牌，你就只是商品。",
     "base": "_valp_v20p1_base_answers.json",
     "ext": "_valp_v20_ext_WCAND11_answers.json",
     "gemini_wr": 0.54, "claude_wr": 0.57, "case_type": "gemini_low_claude_high"},
]

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


def judge_batch(problems, ans_base, ans_ext, client):
    def one(p):
        pid = p["problem_id"]
        ba, ea = ans_base.get(pid), ans_ext.get(pid)
        if not ba or not ea:
            return pid, "missing"
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = ea, ba, "A"
        else:
            left, right, ext_was = ba, ea, "B"
        prompt = JUDGE_PROMPT.format(problem=p["description"],
                                      answer_a=left, answer_b=right)
        try:
            r = client.generate(prompt, max_tokens=400, temperature=0.0)
            v = parse_json_from_llm(r["text"])
        except Exception as e:
            return pid, f"err:{e}"[:30]
        w = v.get("winner", "tie")
        if w == "tie": return pid, "tie"
        return pid, ("ext" if w == ext_was else "base")
    c = {"ext": 0, "base": 0, "tie": 0, "missing": 0, "error": 0}
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(one, p) for p in problems]
        for f in as_completed(futs):
            _pid, res = f.result()
            if res.startswith("err"):
                c["error"] += 1
            else:
                c[res] = c.get(res, 0) + 1
    tot = c["ext"] + c["base"]
    return {**c, "wr_ext": tot and c["ext"] / tot or 0.5}


def main():
    problems = json.loads((CACHE / HOLDOUT_SAMPLE).read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]

    gpt = GPT5Client()
    print(f"Third-family judge: {gpt.model}\n")

    results = []
    for cand in KEEPS:
        print(f"=== [{cand['cid']}/{cand.get('wid','----')}] {cand['aphorism']}  "
              f"[{cand['case_type']}] ===")
        ans_base = json.loads((CACHE / "answers" / cand["base"]).read_text(encoding="utf-8"))
        ans_ext = json.loads((CACHE / "answers" / cand["ext"]).read_text(encoding="utf-8"))
        t0 = time.time()
        res = judge_batch(problems, ans_base, ans_ext, gpt)
        dt = time.time() - t0
        # Diagnostic: does GPT-5.4 side with gemini, claude, or neither?
        gw = cand["gemini_wr"]; cw = cand["claude_wr"]; gpw = res["wr_ext"]
        if gpw >= 0.60 and cw < 0.55:
            diag = "SIDES_WITH_GEMINI (claude is strict)"
        elif gpw < 0.55 and gw >= 0.60:
            diag = "SIDES_WITH_CLAUDE (gemini is loose)"
        elif 0.55 <= gpw < 0.60:
            diag = "IN_BETWEEN (no family consensus)"
        else:
            diag = "UNCLEAR"
        print(f"  GPT-5.4: ext={res['ext']} base={res['base']} tie={res['tie']} "
              f"wr={gpw:.2f} ({dt:.0f}s)")
        print(f"  Gemini={gw:.2f}  Claude={cw:.2f}  GPT-5.4={gpw:.2f}  → {diag}\n")
        results.append({
            "wid": cand.get("wid"), "cid": cand["cid"],
            "aphorism": cand["aphorism"],
            "case_type": cand["case_type"],
            "gemini_wr": gw, "claude_wr": cw, "gpt54_wr": gpw,
            "diagnostic": diag,
            "summary": res,
        })

    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "judge": gpt.model, "results": results}
    prev = json.loads(OUT_LOG.read_text()) if OUT_LOG.exists() else []
    prev.append(log)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"Saved → {OUT_LOG.name}\n")

    print("=== SUMMARY ===")
    print(f"{'cid':10s} {'gemini':8s} {'claude':8s} {'gpt54':8s}  verdict")
    print("-" * 65)
    sides_claude = sides_gemini = between = 0
    for r in results:
        print(f"  {r['cid']:8s} {r['gemini_wr']:<8.2f} {r['claude_wr']:<8.2f} "
              f"{r['gpt54_wr']:<8.2f}  {r['diagnostic']}")
        if "CLAUDE" in r["diagnostic"]: sides_claude += 1
        elif "GEMINI" in r["diagnostic"]: sides_gemini += 1
        else: between += 1
    print(f"\n  Third family sides with Claude: {sides_claude}/{len(results)}")
    print(f"  Third family sides with Gemini: {sides_gemini}/{len(results)}")
    print(f"  In between / unclear:           {between}/{len(results)}")
    # majority-vote gate: candidate passes if ≥2/3 families have wr >=0.55
    print(f"\n  Majority-of-3 judge verdict (≥ 2/3 families ≥ 0.55):")
    for r in results:
        votes = sum(1 for v in (r["gemini_wr"], r["claude_wr"], r["gpt54_wr"]) if v >= 0.55)
        passes = votes >= 2
        print(f"    {r['cid']:9s} votes={votes}/3  {'PASS' if passes else 'FAIL'}")


if __name__ == "__main__":
    main()
