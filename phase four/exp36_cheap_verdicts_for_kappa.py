"""Exp 36 — Cheap-tier per-pid verdicts on the same 4 candidates.

Exp 35 gives us per-pid verdicts from 2 expensive judges (Opus, GPT-5.4)
on W076/W077/W078/WCAND01 × 50 pids. For a Cohen's κ cross-tier
analysis we also need 3 cheap judges (gemini, haiku, gpt-mini) on
the same (pid, candidate) pairs, same side-randomization.

Budget: 4 candidates × 50 pids × 3 cheap judges = 600 calls.
Cost: ~$5. Time: ~5 min at 6 workers.

Output: exp36_cheap_verdicts_log.json with the same
(cand_id -> family -> {pid: verdict}) structure as Exp 35.
"""

import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import cheap_panel
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp36_cheap_verdicts_log.json"

PARALLEL = 6

CANDIDATES = [
    ("_exp10_v20_ext_WCAND05_answers.json", "W076"),
    ("_exp10_v20_ext_WCAND10_answers.json", "W077"),
    ("_exp10_v20_ext_WCROSSL01_answers.json", "W078"),
    ("_validation_v20_ext_WCAND01_answers.json", "WCAND01"),
]

JUDGE_PROMPT = """方法论评审.

## 问题
{problem}

## 解答 A
{answer_a}

## 解答 B
{answer_b}

Score reframe, substance, citability, risk. Decide winner.
Output JSON: {{"winner": "A"/"B"/"tie", "score_a": 1-10, "score_b": 1-10,
  "reasoning": "80 chars max"}}
"""


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception as e:
        return f"err:{str(e)[:30]}"


def load_problems():
    pid_to_prob = {}
    for f in (PROJECT / "phase zero" / "benchmark" / "problems").glob("*.json"):
        for q in json.loads(f.read_text()):
            pid_to_prob[q["problem_id"]] = q.get("description") or q.get("problem") or ""
    return pid_to_prob


def judge_cand(cand_tuple, judges, base_ans, pid_to_prob):
    ext_file, cand_id = cand_tuple
    ext_path = ANS / ext_file
    if not ext_path.exists():
        return None
    ext_ans = json.loads(ext_path.read_text())
    common = sorted(set(base_ans) & set(ext_ans) & set(pid_to_prob))
    print(f"  {cand_id}: {len(common)} pids × {len(judges)} judges = "
          f"{len(common)*len(judges)} calls")

    def judge_pair(judge, pid):
        b = base_ans[pid]; e = ext_ans[pid]
        rng = random.Random(hash(pid) % (2**32))
        if rng.random() < 0.5:
            left, right, ext_was = e, b, "A"
        else:
            left, right, ext_was = b, e, "B"
        w = judge_one(judge, pid_to_prob[pid], left, right)
        if w == "tie": v = "tie"
        elif w in ("A", "B"): v = "ext" if w == ext_was else "base"
        else: v = "err"
        return judge.family, pid, v

    verdicts = {j.family: {} for j in judges}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_pair, j, pid) for j in judges for pid in common]
        for f in as_completed(futs):
            fam, pid, v = f.result()
            verdicts[fam][pid] = v
    print(f"  done in {time.time()-t0:.0f}s")
    return {"cand_id": cand_id, "n_pids": len(common), "verdicts": verdicts}


def main():
    base_ans = json.loads((ANS / "_exp10_v20_base_answers.json").read_text())
    pid_to_prob = load_problems()
    judges = cheap_panel()
    print(f"Cheap judges: {[j.model for j in judges]}\n")

    results = []
    for cand in CANDIDATES:
        print(f"\n=== {cand[1]} ===")
        r = judge_cand(cand, judges, base_ans, pid_to_prob)
        if not r: continue
        # print wr per family
        for fam, v in r["verdicts"].items():
            ne = sum(1 for x in v.values() if x == "ext")
            nb = sum(1 for x in v.values() if x == "base")
            nt = sum(1 for x in v.values() if x == "tie")
            tot = ne + nb
            wr = ne/tot if tot else 0.5
            print(f"  {fam:15s} ext={ne:2d} base={nb:2d} tie={nt:2d}  wr={wr:.2f}")
        results.append(r)

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "judges": [j.model for j in judges],
           "results": results}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
