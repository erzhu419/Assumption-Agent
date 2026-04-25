"""Exp 42 — GSM8K standard-benchmark portability extended to W076 + W077.

Exp 32 ran GSM8K only on W078 (the strongest KEEP). Closes review
weakness #7's objective-benchmark angle for the other two KEEPs:
W076 (凡益之道，与时偕行 / change with the times) and
W077 (没有调查，就没有发言权 / no investigation, no right to speak).

Same 30 problems from gsm8k/main test split, same gemini-3-flash
solver, same 3-family cheap judge panel. We measure:
  - wr_ext per family (subjective LLM judging on quality)
  - accuracy improvement (objective: extracts numerical answer
    and matches the GSM8K gold answer)

The accuracy axis is OBJECTIVE — does the wisdom help the model
solve more GSM8K problems correctly? On Exp 32, W078 actively
HARMED accuracy (29/30 vs 30/30 base). We test the same for
W076 and W077.

Cost: 30 x 4 generations + 30 x 2 x 3 judge calls = 300 calls,
~$5, ~15 min.
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

from datasets import load_dataset
from model_router import cheap, cheap_panel
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp42_gsm8k_w076_w077_log.json"

N = 30
PARALLEL = 6


KEEPS_TO_TEST = [
    {"id": "W076", "aphorism": "凡益之道，与时偕行",
     "source": "《周易》",
     "signal": "环境/认知/工艺正在变迁但旧规则还在沿用",
     "unpacked_for_llm": "判断当前问题是否处在一个旧框架的合理边界正在收窄"
                          "的时刻；如果是，先让框架本身随时代调整，再在新框架"
                          "内做对象级决策。"},
    {"id": "W077", "aphorism": "没有调查，就没有发言权",
     "source": "毛泽东《反对本本主义》",
     "signal": "多因混杂、观察噪声大、没定位清楚就已经要决策时",
     "unpacked_for_llm": "在多因素混杂、观察噪声大的情境里，先建复现/归因"
                          "链路再下结论；调研充分前不要做大决策。"},
]


FRAME_PROMPT = """下面是英文数学题，请用中文给出诊断 frame + 重写。

## 英文原题
{problem}

## 输出 JSON
{{"frame": "object_level / paradigm / hybrid",
  "critical_reframe": "30-60字",
  "rewritten_problem": "100-200字"}}
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
- **必须以 "答案：<number>" 结束**

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

输出 JSON:
{{"winner": "A"/"B"/"tie",
  "a_correct_final": true/false,
  "b_correct_final": true/false,
  "reasoning": "80字内"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


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


def extract_answer(text):
    """Try to extract a numerical answer from the text."""
    # Look for "答案：N" or "answer: N" or final number
    patterns = [r"答案[：:]\s*([+-]?\d[\d,\.]*)", r"answer[:\s]+([+-]?\d[\d,\.]*)",
                r"最终答案[:\s：]*([+-]?\d[\d,\.]*)"]
    for p in patterns:
        m = re.search(p, text)
        if m:
            try:
                return float(m.group(1).replace(",", ""))
            except: pass
    # Fallback: last number in text
    nums = re.findall(r"([+-]?\d[\d,\.]*)", text)
    if nums:
        try: return float(nums[-1].replace(",", ""))
        except: return None
    return None


def gold_value(text):
    """GSM8K gold answer follows '#### N' convention."""
    m = re.search(r"####\s*([+-]?\d[\d,\.]*)", text)
    if m:
        try: return float(m.group(1).replace(",", ""))
        except: return None
    return None


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
    problems = [{"pid": f"gsm8k_{i:04d}", "problem": x["question"], "gold": x["answer"]}
                  for i, x in enumerate(ds)]
    print(f"Loaded {len(problems)} GSM8K problems")

    solver = cheap("gemini")
    judges = cheap_panel()
    print(f"Solver: {solver.model}; Judges: {[j.model for j in judges]}\n")

    # Generate base + 2 ext (W076, W077). Reuse W078 ext from Exp 32 if present.
    print(f"[1/2] Generating answers for base + W076 ext + W077 ext = {3*len(problems)}...")
    answers = {"base": {}}
    for kp in KEEPS_TO_TEST:
        answers[kp["id"]] = {}

    def gen(p, kind):
        wisdoms = [] if kind == "base" else [next(kp for kp in KEEPS_TO_TEST if kp["id"] == kind)]
        return p["pid"], kind, solve(solver, p["problem"], wisdoms)

    tasks = [(p, "base") for p in problems] + \
            [(p, kp["id"]) for p in problems for kp in KEEPS_TO_TEST]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(gen, p, k) for p, k in tasks]
        for f in as_completed(futs):
            pid, kind, ans = f.result()
            answers[kind][pid] = ans
            done += 1
            if done % 15 == 0:
                print(f"  gen {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Objective accuracy (independent of judges) — does the answer
    # extracted match GSM8K gold?
    print(f"\n[Objective] Per-condition accuracy on GSM8K (extracted-vs-gold):")
    accuracy = {}
    for kind in answers:
        n_correct = 0
        for p in problems:
            ans = answers[kind].get(p["pid"], "")
            if not ans or ans.startswith("[turn"):
                continue
            extracted = extract_answer(ans)
            gold = gold_value(p["gold"])
            if extracted is not None and gold is not None and abs(extracted - gold) < 0.01:
                n_correct += 1
        accuracy[kind] = (n_correct, len(problems))
        print(f"  {kind}: {n_correct}/{len(problems)} = {n_correct/len(problems):.2f}")

    # Subjective judging (3 families x 2 KEEPs x 30 problems)
    print(f"\n[2/2] Judging {len(KEEPS_TO_TEST)*len(problems)*len(judges)} pairs with 3 cheap judges")
    verdicts = {kp["id"]: {j.family: {} for j in judges} for kp in KEEPS_TO_TEST}
    correctness_judged = {kp["id"]: {j.family: {"base_correct": 0, "ext_correct": 0}
                                       for j in judges} for kp in KEEPS_TO_TEST}

    def jt(judge, kp, p):
        pid = p["pid"]
        b = answers["base"].get(pid, "")
        e = answers[kp["id"]].get(pid, "")
        if not b or not e or b.startswith("[") or e.startswith("["):
            return judge.family, kp["id"], pid, "missing", (False, False)
        rng = random.Random((hash(pid) ^ hash(kp["id"])) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w, a_corr, b_corr = judge_one(judge, p["problem"], p["gold"], left, right)
        if ext_was == "A":
            ext_correct = a_corr; base_correct = b_corr
        else:
            ext_correct = b_corr; base_correct = a_corr
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return judge.family, kp["id"], pid, v, (base_correct, ext_correct)

    jtasks = [(j, kp, p) for j in judges for kp in KEEPS_TO_TEST for p in problems]
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(jt, *t) for t in jtasks]
        for f in as_completed(futs):
            fam, kid, pid, v, (bc, ec) = f.result()
            verdicts[kid][fam][pid] = v
            if bc: correctness_judged[kid][fam]["base_correct"] += 1
            if ec: correctness_judged[kid][fam]["ext_correct"] += 1
            done += 1
            if done % 30 == 0:
                print(f"  judge {done}/{len(jtasks)} ({time.time()-t0:.0f}s)")

    # Report
    print(f"\n=== Per-family verdicts on 30 GSM8K problems ===")
    for kid in (kp["id"] for kp in KEEPS_TO_TEST):
        print(f"\n  {kid}:")
        for fam in (j.family for j in judges):
            v = verdicts[kid][fam]
            ne = sum(1 for x in v.values() if x == "ext")
            nb = sum(1 for x in v.values() if x == "base")
            nt = sum(1 for x in v.values() if x == "tie")
            tot = ne + nb
            wr = ne / tot if tot else 0.5
            cor = correctness_judged[kid][fam]
            print(f"    {fam:14s} ext={ne} base={nb} tie={nt}  wr={wr:.2f}  "
                  f"acc(judge): base={cor['base_correct']}/{len(v)}, ext={cor['ext_correct']}/{len(v)}")

    print(f"\n=== Headline (objective accuracy, independent of LLM judges) ===")
    print(f"  base   : {accuracy['base'][0]}/{accuracy['base'][1]}")
    for kp in KEEPS_TO_TEST:
        kid = kp["id"]
        delta = accuracy[kid][0] - accuracy["base"][0]
        sign = "+" if delta > 0 else ""
        verdict = ("HARMS" if delta < -1 else "HELPS" if delta > 1 else "neutral")
        print(f"  {kid} ext: {accuracy[kid][0]}/{accuracy[kid][1]}  delta = {sign}{delta}  ({verdict})")
    print(f"\n  Compare to Exp 32: W078 ext 29/30 vs base 30/30 (HARMS, delta = -1)")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "benchmark": "GSM8K test split (first 30)",
           "wisdoms_tested": [kp["id"] for kp in KEEPS_TO_TEST],
           "solver": solver.model,
           "judges": [j.model for j in judges],
           "n_problems": len(problems),
           "objective_accuracy": accuracy,
           "verdicts_per_family": verdicts,
           "correctness_judged": correctness_judged}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
