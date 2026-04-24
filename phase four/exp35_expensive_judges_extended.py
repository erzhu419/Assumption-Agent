"""Exp 35 — Extended expensive-judge audit on more KEEPs + cached ext.

Extends Exp 34 (W078 only) to W076, W077, WCAND01 — all the candidates
with 50-pid cached base/ext answer pairs we can judge without
regenerating answers.

Cached (base, ext) on 50 holdout-like pids for:
  W076 (WCAND05), W077 (WCAND10), W078 (WCROSSL01), WCAND01
  WCAND02 is only 10 pids (skip or run cautiously)

Budget: ~400 expensive API calls (4 candidates × 50 pids × 2 judges,
minus W078 already done = 300 new calls). ~$25, ~10 min at 4
workers.

Output: per-candidate × per-expensive-judge wr_ext + Wilson CI, plus
full (pid -> winner) verdicts for downstream Cohen's κ analysis.
"""

import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import expensive
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp35_expensive_extended_log.json"

PARALLEL = 4

# candidate -> (ext_answers_filename, cand_id_shown_in_paper)
CANDIDATES = [
    ("_exp10_v20_ext_WCAND05_answers.json", "W076"),
    ("_exp10_v20_ext_WCAND10_answers.json", "W077"),
    ("_exp10_v20_ext_WCROSSL01_answers.json", "W078"),
    ("_validation_v20_ext_WCAND01_answers.json", "WCAND01"),
]

JUDGE_PROMPT = """方法论评审 (Methodology review).

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


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception as e:
        return f"err:{str(e)[:40]}"


def load_problems():
    """Load {pid -> problem} from the benchmark pool."""
    pid_to_prob = {}
    for f in (PROJECT / "phase zero" / "benchmark" / "problems").glob("*.json"):
        for q in json.loads(f.read_text()):
            pid_to_prob[q["problem_id"]] = q.get("description") or q.get("problem") or ""
    return pid_to_prob


def judge_cand(cand_tuple, judges, base_ans, pid_to_prob, existing_log):
    ext_file, cand_id = cand_tuple
    ext_path = ANS / ext_file
    if not ext_path.exists():
        print(f"  SKIP {cand_id}: {ext_file} missing")
        return None
    ext_ans = json.loads(ext_path.read_text())
    common = sorted(set(base_ans) & set(ext_ans) & set(pid_to_prob))
    print(f"  {cand_id}: {len(common)} pids to judge")

    def judge_pair(judge, pid):
        # Reuse cached W078 Exp34 verdicts if already done
        if cand_id == "W078" and judge.family in existing_log.get("verdicts", {}) \
                and pid in existing_log["verdicts"][judge.family]:
            return judge.family, pid, existing_log["verdicts"][judge.family][pid]
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
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_pair, j, pid) for j in judges for pid in common]
        for f in as_completed(futs):
            fam, pid, v = f.result()
            verdicts[fam][pid] = v
    return {"cand_id": cand_id, "n_pids": len(common), "verdicts": verdicts}


def main():
    base_ans = json.loads((ANS / "_exp10_v20_base_answers.json").read_text())
    pid_to_prob = load_problems()
    print(f"Base answers: {len(base_ans)} pids")

    # Try to reuse Exp 34 W078 verdicts to save 100 API calls
    existing = cache_load(AUTO_DIR / "exp34_expensive_judges_log.json", default={})
    if existing.get("verdicts"):
        print("Reusing Exp 34 W078 expensive verdicts")

    judges = [expensive(n) for n in ("claude_opus", "gpt5")]
    print(f"Expensive judges: {[j.model for j in judges]}\n")

    all_results = []
    for cand in CANDIDATES:
        print(f"\n=== {cand[1]} ===")
        t0 = time.time()
        r = judge_cand(cand, judges, base_ans, pid_to_prob, existing)
        if not r:
            continue
        for fam, verd in r["verdicts"].items():
            ne = sum(1 for v in verd.values() if v == "ext")
            nb = sum(1 for v in verd.values() if v == "base")
            nt = sum(1 for v in verd.values() if v == "tie")
            tot = ne + nb
            wr = ne / tot if tot else 0.5
            # Wilson CI
            try:
                from scipy.stats import binomtest
                ci = binomtest(ne, tot).proportion_ci(method="wilson")
                ci_str = f"[{ci.low:.2f}, {ci.high:.2f}]"
            except:
                ci_str = "(?)"
            print(f"  {fam:15s} ext={ne:2d} base={nb:2d} tie={nt:2d}  "
                  f"wr={wr:.2f}  95% CI {ci_str}")
        r["wall_time_sec"] = time.time() - t0
        all_results.append(r)

    # Summary
    print(f"\n=== Extended expensive-judge summary ===")
    print(f"{'cand':10s}  {'opus':>12s}  {'gpt-5.4':>12s}")
    for r in all_results:
        parts = [f"{r['cand_id']:10s}"]
        for fam in ("claude_opus", "gpt5"):
            v = r["verdicts"].get(fam, {})
            ne = sum(1 for x in v.values() if x == "ext")
            nb = sum(1 for x in v.values() if x == "base")
            tot = ne + nb
            wr = ne / tot if tot else 0.5
            parts.append(f"{wr:>12.2f}")
        print("  ".join(parts))

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "judges": [j.model for j in judges],
           "n_candidates": len(all_results),
           "results": all_results}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
