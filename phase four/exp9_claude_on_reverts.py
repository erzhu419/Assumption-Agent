"""Exp 9: Claude Opus re-judges the 9 REVERT candidates.

Question: under the new cross-family gate, would any REVERT actually
pass (claude_wr >= 0.55)? Either answer is informative:
  - All < 0.55  → new gate is consistent with old gate on REVERTs
  - Some >= 0.55 → old gate had FALSE NEGATIVES (wrongly rejected)
                   that cross-family would have saved.

Uses same 50 held-out pairs as Exp 1, same JUDGE_PROMPT, same per-PID
side-randomization rule. The only delta from validate_parallel:
  - Judge is claude-opus-4-6 instead of gemini-3-flash
  - Target list is the 9 REVERT candidates instead of the 3 KEEPs
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
sys.path.insert(0, str(PROJECT / "phase four"))

from claude_proxy_client import ClaudeProxyClient
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp9_claude_reverts_log.json"
HOLDOUT_SAMPLE = "sample_holdout_50.json"
PARALLEL = 6

# Nine REVERT candidates, sorted by original gemini wr (descending)
REVERTS = [
    {"cid": "WCAND01", "gemini_wr": 0.58, "aphorism": "上工治未病，不治已病"},
    {"cid": "WCAND06", "gemini_wr": 0.58, "aphorism": "覆水难收，向前算账"},
    {"cid": "WCAND03", "gemini_wr": 0.56, "aphorism": "凡事预则立，不预则废"},
    {"cid": "WCAND07", "gemini_wr": 0.56, "aphorism": "亲兄弟，明算账"},
    {"cid": "WCAND09", "gemini_wr": 0.55, "aphorism": "不谋全局者，不足谋一域"},
    {"cid": "WCAND11", "gemini_wr": 0.54, "aphorism": "若不是品牌，你就只是商品。"},
    {"cid": "WCAND08", "gemini_wr": 0.51, "aphorism": "想理解行为，先看激励"},
    {"cid": "WCAND04", "gemini_wr": 0.48, "aphorism": "急则治其标，缓则治其本"},
    {"cid": "WCAND02", "gemini_wr": 0.46, "aphorism": "别高效解决一个被看错的问题"},
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


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, answer_a=a,
                                                  answer_b=b),
                            max_tokens=400, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception as e:
        return f"err:{e}"[:30]


def claude_judge_batch(problems, ans_base, ans_ext, client):
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
        w = judge_one(client, p["description"], left, right)
        if w == "tie":
            return pid, "tie"
        if w in ("A", "B"):
            return pid, ("ext" if w == ext_was else "base")
        return pid, w  # error
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
    wr = c["ext"] / tot if tot else 0.5
    return {**c, "wr_claude": wr}


def main():
    problems = json.loads((CACHE / HOLDOUT_SAMPLE).read_text(encoding="utf-8"))
    problems = [p for p in problems if "description" in p]

    claude = ClaudeProxyClient()
    print(f"Judge: {claude.model}\n")

    results = []
    t_total = time.time()
    for cand in REVERTS:
        cid = cand["cid"]
        print(f"=== [{cid}] {cand['aphorism']} (gemini={cand['gemini_wr']}) ===")
        base_path = CACHE / "answers" / "_valp_v20p1_base_answers.json"
        ext_path = CACHE / "answers" / f"_valp_v20_ext_{cid}_answers.json"
        if not base_path.exists() or not ext_path.exists():
            print(f"  [SKIP] missing answer files"); continue
        ans_base = json.loads(base_path.read_text(encoding="utf-8"))
        ans_ext = json.loads(ext_path.read_text(encoding="utf-8"))
        t0 = time.time()
        res = claude_judge_batch(problems, ans_base, ans_ext, claude)
        dt = time.time() - t0
        pass_new_gate = (cand["gemini_wr"] >= 0.60 and res["wr_claude"] >= 0.55)
        label = "WOULD_KEEP_UNDER_NEW_GATE" if pass_new_gate else "STILL_REVERT"
        print(f"  Claude: ext={res['ext']} base={res['base']} tie={res['tie']} "
              f"wr_claude={res['wr_claude']:.2f} ({dt:.0f}s)  → {label}\n")
        results.append({
            "cid": cid, "aphorism": cand["aphorism"],
            "original_gemini_wr": cand["gemini_wr"],
            "claude_wr": res["wr_claude"],
            "ext_wins": res["ext"], "base_wins": res["base"], "ties": res["tie"],
            "would_pass_new_gate": pass_new_gate,
        })

    # Save log
    log = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "judge": claude.model, "n_candidates": len(results),
           "total_wall_sec": int(time.time() - t_total),
           "results": results}
    prev = json.loads(OUT_LOG.read_text()) if OUT_LOG.exists() else []
    prev.append(log)
    OUT_LOG.write_text(json.dumps(prev, ensure_ascii=False, indent=2))
    print(f"Saved → {OUT_LOG.name}\n")

    # Summary
    print(f"=== SUMMARY ===")
    print(f"{'cid':10s} {'gemini':8s} {'claude':8s}  fate under new gate")
    print("-" * 60)
    would_keep = 0
    for r in results:
        cw = r["claude_wr"]; gw = r["original_gemini_wr"]
        fate = "WOULD_KEEP" if r["would_pass_new_gate"] else "STILL_REVERT"
        print(f"  {r['cid']:8s} {gw:<8.2f} {cw:<8.2f}  {fate}")
        if r["would_pass_new_gate"]: would_keep += 1
    print(f"\n  Of {len(results)} original REVERTs, {would_keep} would pass the "
          f"new cross-family gate (gemini>=0.60 AND claude>=0.55)")
    if would_keep == 0:
        print("  → new gate is CONSISTENT with old on REVERTs (no false-negative "
              "rescue)")
    else:
        print("  → new gate has DISAGREEMENTS; some original REVERTs would now pass")


if __name__ == "__main__":
    main()
