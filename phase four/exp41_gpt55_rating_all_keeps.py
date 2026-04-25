"""Exp 41 — Extend gpt-5.5 structured rating from W078 only to all 3 KEEPs.

Closes review weakness #7 (no human ground truth) partially:
the existing Exp 38 ran only on W078. Same protocol now on W076
and W077. Same 30-pid sample, same 1-5 Likert rubric across 4
dimensions (substantive, methodological, fit, prescriptive).

Cost: 60 calls (2 KEEPs * 30 pids), ~$10, ~5 min.
"""

import json
import random
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

from dotenv import load_dotenv
for p in (PROJECT / ".env", PROJECT / "phase zero" / ".env"):
    if p.exists(): load_dotenv(p)
import os
from openai import OpenAI
from llm_client import parse_json_from_llm

CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp41_gpt55_rating_all_keeps_log.json"

PARALLEL = 4
N_PIDS = 30


class _GPT55Client:
    def __init__(self):
        self.model = "gpt-5.5"
        self.family = "gpt55"
        base = os.environ.get("GPT5_BASE_URL", "https://ruoli.dev/v1")
        key = os.environ.get("GPT5_API_KEY", "")
        self._client = OpenAI(base_url=base, api_key=key)

    def generate(self, prompt, max_tokens=2000, temperature=0.0):
        r = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return {"text": r.choices[0].message.content or ""}


KEEPS = [
    {"id": "W076", "ext_file": "_exp10_v20_ext_WCAND05_answers.json"},
    {"id": "W077", "ext_file": "_exp10_v20_ext_WCAND10_answers.json"},
]


RATING_PROMPT = """You are a careful methodological reviewer evaluating two
answers to the SAME problem. The two answers were produced by the same
solver model under two different scaffolding configurations. You do NOT
know which configuration produced which answer; do not speculate.

## Problem
{problem}

## Answer A
{answer_a}

## Answer B
{answer_b}

For each answer, give an independent 1-5 Likert score on FOUR
dimensions. Reason briefly (under 60 words). Then output JSON.

1. **substantive_content** (1-5): Does the answer's actual claims address the problem?
2. **methodological_soundness** (1-5): Is the reasoning structure valid?
3. **problem_fit** (1-5): Does the answer engage what was actually asked?
4. **prescriptive_burden** (1-5, REVERSED): higher = better calibrated, less over-prescribing.

Output JSON only:
{{
  "reasoning_a": "...",
  "scores_a": {{"substantive": 1-5, "method": 1-5, "fit": 1-5, "prescriptive": 1-5}},
  "reasoning_b": "...",
  "scores_b": {{"substantive": 1-5, "method": 1-5, "fit": 1-5, "prescriptive": 1-5}}
}}
"""


def cache_load(p, default=None):
    try: return json.loads(Path(p).read_text(encoding="utf-8"))
    except: return default


def load_problems():
    pid_to_prob = {}
    for f in (PROJECT / "phase zero" / "benchmark" / "problems").glob("*.json"):
        for q in json.loads(f.read_text()):
            pid_to_prob[q["problem_id"]] = q.get("description") or q.get("problem") or ""
    return pid_to_prob


def rate_pair(client, problem, base, ext, pid):
    rng = random.Random(hash(pid) % (2**32))
    if rng.random() < 0.5:
        a, b, ext_was = ext, base, "A"
    else:
        a, b, ext_was = base, ext, "B"
    try:
        r = client.generate(RATING_PROMPT.format(problem=problem, answer_a=a, answer_b=b),
                            max_tokens=1500, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        sa = v.get("scores_a", {}); sb = v.get("scores_b", {})
        if ext_was == "A":
            ext_scores, base_scores = sa, sb
        else:
            ext_scores, base_scores = sb, sa
        return {"pid": pid, "ext_was": ext_was,
                "ext_scores": ext_scores, "base_scores": base_scores,
                "reasoning_ext": v.get("reasoning_a" if ext_was == "A" else "reasoning_b", "")[:300]}
    except Exception as e:
        return {"pid": pid, "ext_was": ext_was, "error": str(e)[:80]}


def run_one_keep(client, kp, pid_to_prob, base_ans, sample_pids):
    print(f"\n=== {kp['id']} ===")
    ext_ans = json.loads((ANS / kp["ext_file"]).read_text())
    common = sorted(set(base_ans) & set(ext_ans) & set(pid_to_prob))
    sample = [p for p in sample_pids if p in common][:N_PIDS]
    print(f"  rating {len(sample)} (pid, base, ext) triples")

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(rate_pair, client, pid_to_prob[p],
                            base_ans[p], ext_ans[p], p) for p in sample]
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            n = len(results)
            if n % 5 == 0:
                print(f"    {n}/{len(sample)} ({time.time()-t0:.0f}s)")

    dims = ["substantive", "method", "fit", "prescriptive"]
    valid = [r for r in results if "error" not in r and r.get("ext_scores") and r.get("base_scores")]
    print(f"  Valid: {len(valid)}/{len(results)}")

    summary = {"valid": len(valid), "n_total": len(results), "per_dim": {}}
    for d in dims:
        ext_vals = [r["ext_scores"].get(d) for r in valid if r["ext_scores"].get(d) is not None]
        base_vals = [r["base_scores"].get(d) for r in valid if r["base_scores"].get(d) is not None]
        diffs = [r["ext_scores"][d] - r["base_scores"][d]
                  for r in valid if r["ext_scores"].get(d) and r["base_scores"].get(d)]
        n_ext_better = sum(1 for delta in diffs if delta > 0)
        em = sum(ext_vals) / len(ext_vals) if ext_vals else 0
        bm = sum(base_vals) / len(base_vals) if base_vals else 0
        dm = sum(diffs) / len(diffs) if diffs else 0
        summary["per_dim"][d] = {"ext_mean": em, "base_mean": bm,
                                    "delta": dm, "n_ext_better": n_ext_better,
                                    "n": len(diffs)}
        print(f"    {d:14s}  ext={em:.2f} base={bm:.2f} delta={dm:+.2f} ext>base={n_ext_better}/{len(diffs)}")

    # Combined
    n_ext_combined = n_base_combined = n_tie_combined = 0
    for r in valid:
        try:
            ext_avg = sum(r["ext_scores"][d] for d in dims) / 4
            base_avg = sum(r["base_scores"][d] for d in dims) / 4
            if ext_avg > base_avg + 0.1: n_ext_combined += 1
            elif base_avg > ext_avg + 0.1: n_base_combined += 1
            else: n_tie_combined += 1
        except: pass
    summary["combined"] = {"ext_better": n_ext_combined,
                            "base_better": n_base_combined,
                            "tie": n_tie_combined,
                            "n": len(valid)}
    print(f"  Combined 4-dim mean: ext_better={n_ext_combined} base_better={n_base_combined} tie={n_tie_combined}")
    return {"keep_id": kp["id"], "summary": summary, "results": results}


def main():
    base_ans = json.loads((ANS / "_exp10_v20_base_answers.json").read_text())
    pid_to_prob = load_problems()
    common = sorted(set(base_ans) & set(pid_to_prob))
    rng = random.Random(42)
    sample_pids = rng.sample(common, min(N_PIDS, len(common)))
    print(f"Sample pids ({len(sample_pids)}): seed=42")

    client = _GPT55Client()

    all_results = []
    for kp in KEEPS:
        all_results.append(run_one_keep(client, kp, pid_to_prob, base_ans, sample_pids))

    # Write log; merge with existing exp38 W078 result for completeness
    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rater_model": "gpt-5.5",
           "rater_role": "structured Likert (1-5) on 4 dimensions",
           "n_pids_sampled": len(sample_pids),
           "results_per_keep": all_results}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
