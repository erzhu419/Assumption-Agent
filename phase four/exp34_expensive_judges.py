"""Exp 34 — Expensive-tier judge sanity check.

Reviewer objection: the audit stack's three judges (gemini-3-flash,
claude-haiku, gpt-5.4-mini) are all in the cheap tier. Does the null
verdict hold when we upgrade to the expensive tier (claude-opus-4.6,
gpt-5.4)?

Re-judge the cached Exp 10 pairs for W078 (WCROSSL01) — the strongest
survivor, the only one that made it through majority-of-3 at n=50 —
with expensive-tier judges on the same 50 holdout pids.

If expensive judges also give wr_ext < 0.55, the null result is
robust to judge budget. If they give wr_ext >= 0.60, the audit stack
missed a real signal visible only to stronger raters — which would
itself be a publishable finding.

Cost envelope: 50 pids x 2 judges = 100 judge calls. Opus ~$0.08/call,
GPT-5.4 ~$0.05/call; ~$6.5 total. Worth it for a headline sanity check.
"""

import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from model_router import expensive, EXPENSIVE_MODELS
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp34_expensive_judges_log.json"

PARALLEL = 4  # expensive calls — don't hammer

JUDGE_PROMPT = """方法论评审 (Methodology review).

## 问题 (Problem)
{problem}

## 解答 A (Answer A)
{answer_a}

## 解答 B (Answer B)
{answer_b}

Score four dimensions: reframe, substance, citability, risk. Decide winner.

Output JSON:
{{"winner": "A"/"B"/"tie",
  "score_a": 1-10, "score_b": 1-10,
  "reasoning": "80 characters or fewer"}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def judge_one(client, problem, a, b):
    try:
        r = client.generate(JUDGE_PROMPT.format(problem=problem, answer_a=a,
                                                  answer_b=b),
                            max_tokens=300, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        return v.get("winner", "tie")
    except Exception as e:
        return f"err:{str(e)[:40]}"


def main():
    # Load base/ext answers
    base_ans = cache_load(ANS / "_exp10_v20_base_answers.json", default={})
    ext_ans = cache_load(ANS / "_exp10_v20_ext_WCROSSL01_answers.json",
                          default={})
    # Problems: the Exp 10 pids span the benchmark pool — look them up there
    pid_to_prob = {}
    pool_dir = PROJECT / "phase zero" / "benchmark" / "problems"
    for f in pool_dir.glob("*.json"):
        for q in json.loads(f.read_text()):
            pid_to_prob[q["problem_id"]] = q.get("description") or q.get("problem") or ""
    common = sorted(set(base_ans) & set(ext_ans) & set(pid_to_prob))
    print(f"Loaded {len(common)} common pids (base∩ext∩benchmark-pool)")

    judges = [expensive(n) for n in ("claude_opus", "gpt5")]
    print(f"Expensive judges: {[j.model for j in judges]}\n")

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
        return judge.family, pid, v, w

    print(f"[1/1] Running {len(judges)} × {len(common)} = {len(judges)*len(common)} judgements...")
    t0 = time.time()
    verdicts = {j.family: {} for j in judges}
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(judge_pair, j, pid)
                 for j in judges for pid in common]
        done = 0
        for f in as_completed(futs):
            fam, pid, v, raw = f.result()
            verdicts[fam][pid] = v
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(futs)} ({time.time()-t0:.0f}s)")

    print(f"\n=== Expensive-tier judge verdicts on W078 × 50 holdout pids ===")
    wrs = {}
    for fam, verd in verdicts.items():
        n_ext = sum(1 for v in verd.values() if v == "ext")
        n_base = sum(1 for v in verd.values() if v == "base")
        n_tie = sum(1 for v in verd.values() if v == "tie")
        n_err = sum(1 for v in verd.values() if v == "err")
        tot = n_ext + n_base
        wr = n_ext / tot if tot else 0.5
        wrs[fam] = wr
        print(f"  {fam:15s} ext={n_ext} base={n_base} tie={n_tie} err={n_err}  "
              f"wr_ext={wr:.2f}")

    mean_wr = sum(wrs.values()) / len(wrs)
    print(f"\n  Mean wr_ext across 2 expensive families: {mean_wr:.2f}")
    print(f"  Threshold wr_ext >= 0.55 for signal? "
          f"{'YES' if mean_wr >= 0.55 else 'NO'}")
    print(f"  Majority-of-2 agreement on 'ext wins'? "
          f"{sum(1 for pid in common if sum(verdicts[j.family][pid] == 'ext' for j in judges) >= 2)}/{len(common)}")

    # Compare with cheap-tier historical wr (from exp10_multijudge cross-family)
    print(f"\n  Historical cheap-tier on same pairs (from paper §4.14):")
    print(f"    gemini (inner-loop)    : wr_ext = 0.60")
    print(f"    claude-haiku (L1)      : wr_ext = 0.51")
    print(f"    gpt-5.4-mini (L1)      : wr_ext = 0.57 (approx)")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "target_wisdom": "W078 (WCROSSL01, 是骡子是马，拉出来遛遛)",
           "n_pids": len(common),
           "judge_tier": "expensive",
           "judges": [j.model for j in judges],
           "per_family_wr": wrs,
           "mean_wr_ext": mean_wr,
           "signal_under_expensive_judges": mean_wr >= 0.55,
           "verdicts": verdicts}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
