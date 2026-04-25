"""Exp 32 — Standard-benchmark portability: GSM8K.

Addresses reviewer objection #4 (no standard benchmark) with real
HuggingFace GSM8K — not our own problem pool.

30 problems from gsm8k/main test split. For each:
  - Generate v20-lite base + ext(+W078) with gemini-3-flash solver
  - Judge with 3 families
  - Report per-family wr_ext + whether signal replicates

Compare to Exp 31 (our math pool): if both null, strong cross-benchmark
confirmation. If GSM8K positive but our math null, opposite-distribution
finding (our pool has problems unlike GSM8K).

GSM8K problems are English math word problems. v20 scaffold was designed
for Chinese open-ended problems, so we translate problems to Chinese at
solve time (prompt includes "以下英文问题请用中文作答" directive).
"""

import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from datasets import load_dataset
from model_router import cheap, cheap_panel
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp32_gsm8k_log.json"

N = 30
PARALLEL = 6


FRAME_PROMPT = """下面是英文数学题，请用中文给出诊断 frame + 重写。

## 英文原题
{problem}

## 输出 JSON
{{"frame": "object_level / paradigm / hybrid",
  "critical_reframe": "一句话中文解释这题真正在考什么 (30-60字)",
  "rewritten_problem": "中文重写版 (100-200字)，显式化数学结构"}}
"""

EXECUTE_PROMPT = """# 解决数学题

## PRIMARY FRAME
- frame: {frame}
- critical reframe: {critical_reframe}

## 题（重写）
{rewritten_problem}

## 次要参考 wisdom
{wisdom_block}

## 要求
- 中文作答，给出解题思路 + 最终数值答案
- ≤ 400 字

开始："""

JUDGE_PROMPT = """数学解题评审。

## 问题（英文原题）
{problem}

## 参考答案
{gold}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

评审：最终数值是否正确？解题过程是否清晰？

输出 JSON:
{{"winner": "A"/"B"/"tie",
  "a_correct_final": true/false,
  "b_correct_final": true/false,
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


def solve(client, problem, wisdoms):
    try:
        r = client.generate(FRAME_PROMPT.format(problem=problem),
                            max_tokens=500, temperature=0.2)
        m = parse_json_from_llm(r["text"])
    except Exception as e:
        return f"[turn0 err: {e}]"
    wb = "\n".join(f"• {w['aphorism']}: {w.get('unpacked_for_llm','')[:150]}"
                    for w in wisdoms) if wisdoms else "(无)"
    try:
        r = client.generate(EXECUTE_PROMPT.format(
            frame=m.get("frame", "object_level"),
            critical_reframe=m.get("critical_reframe", ""),
            rewritten_problem=m.get("rewritten_problem", problem),
            wisdom_block=wb), max_tokens=800, temperature=0.2)
        return r["text"].strip()
    except Exception as e:
        return f"[turn1 err: {e}]"


def judge_one(client, problem, gold, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, gold=gold,
                                                  answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return (v.get("winner", "tie"),
                bool(v.get("a_correct_final", False)),
                bool(v.get("b_correct_final", False)))
    except Exception:
        return ("err", False, False)


def main():
    print(f"Loading GSM8K test split (first {N})...")
    ds = load_dataset("gsm8k", "main", split=f"test[:{N}]")
    problems = [{"pid": f"gsm8k_{i:04d}", "problem": x["question"],
                  "gold": x["answer"]} for i, x in enumerate(ds)]
    print(f"Loaded {len(problems)} GSM8K problems")

    w078 = load_w078()
    solver = cheap("gemini")
    judges = cheap_panel()
    print(f"Solver: {solver.model}")
    print(f"Judges: {[j.model for j in judges]}\n")

    # Step 1: generate answers
    print(f"[1/2] Solving {len(problems)} GSM8K problems with/without W078...")
    answers = {"base": {}, "ext": {}}

    def gen(lib_name, p, wisdoms):
        ans = solve(solver, p["problem"], wisdoms)
        return lib_name, p["pid"], ans

    tasks = [("base", p, []) for p in problems] + \
            [("ext", p, [w078]) for p in problems]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen, lib, p, w) for lib, p, w in tasks]
        for f in as_completed(futs):
            lib, pid, ans = f.result()
            answers[lib][pid] = ans
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Step 2: judge by 3 families
    print(f"\n[2/2] Judge 3 families × {len(problems)} pairs...")
    verdicts = {j.family: {} for j in judges}
    correctness = {j.family: {"base_correct": 0, "ext_correct": 0} for j in judges}

    def judge_pair(judge, p):
        pid = p["pid"]
        b = answers["base"].get(pid, "")
        e = answers["ext"].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return judge.family, pid, "missing", (False, False)
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w, a_corr, b_corr = judge_one(judge, p["problem"], p["gold"], left, right)
        # Map back: track base's and ext's correctness
        if ext_was == "A":
            ext_correct = a_corr; base_correct = b_corr
        else:
            ext_correct = b_corr; base_correct = a_corr
        if w == "tie":
            verdict = "tie"
        elif w in ("A", "B"):
            verdict = "ext" if w == ext_was else "base"
        else:
            verdict = "err"
        return judge.family, pid, verdict, (base_correct, ext_correct)

    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_pair, j, p) for j in judges for p in problems]
        for f in as_completed(futs):
            fam, pid, v, (bc, ec) = f.result()
            verdicts[fam][pid] = v
            if bc: correctness[fam]["base_correct"] += 1
            if ec: correctness[fam]["ext_correct"] += 1

    print(f"\n=== Per-family verdicts on 30 GSM8K problems ===")
    wrs = {}
    for fam, verd in verdicts.items():
        ne = sum(1 for v in verd.values() if v == "ext")
        nb = sum(1 for v in verd.values() if v == "base")
        nt = sum(1 for v in verd.values() if v == "tie")
        tot = ne + nb
        wr = ne / tot if tot else 0.5
        wrs[fam] = wr
        cor = correctness[fam]
        print(f"  {fam:15s} ext={ne} base={nb} tie={nt}  wr_ext={wr:.2f}  "
              f"accuracy: base={cor['base_correct']}/{len(verd)}, "
              f"ext={cor['ext_correct']}/{len(verd)}")

    mean_wr = sum(wrs.values()) / len(wrs)
    all_above = all(w >= 0.55 for w in wrs.values())
    print(f"\n  Mean wr_ext across families: {mean_wr:.2f}")
    print(f"  All families >= 0.55? {all_above}")
    print(f"  Signal replicates on GSM8K? {'YES' if all_above and mean_wr >= 0.55 else 'NO'}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "benchmark": "GSM8K test split (first 30)",
           "target_wisdom": "W078",
           "solver": solver.model,
           "judges": [j.model for j in judges],
           "n_problems": len(problems),
           "per_family_wr": wrs, "correctness": correctness,
           "mean_wr_ext": mean_wr,
           "signal_replicates": all_above and mean_wr >= 0.55,
           "verdicts": verdicts}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
