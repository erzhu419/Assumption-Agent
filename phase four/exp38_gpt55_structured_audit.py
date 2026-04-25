"""Exp 38 — GPT-5.5 structured rating as a stronger non-panel anchor.

Real human annotation is logistically expensive. As a substitute we
use gpt-5.5 — the strongest model available on the proxy and
NOT a member of the existing 5-judge audit panel — to give a
multi-dimensional STRUCTURED rating (not just A/B winner) on the
same (base, ext) pairs the panel adjudicated.

The point: separate two failure modes the audit cannot otherwise
distinguish:
  (a) wisdom-augmented answer is genuinely worse on substantive
      content — judges agree it lost
  (b) answers are similar in substantive content; judges disagree
      stylistically

For each (pid, base, ext) we have gpt-5.5 score 4 dimensions on a
1-5 Likert:
  - substantive_content (does the answer's actual claims address the problem)
  - methodological_soundness (is the reasoning structure valid)
  - problem_fit (does the answer engage what was asked)
  - prescriptive_burden (does the answer over-prescribe / over-decide)

These dims are inspired by what a careful human reviewer would do.
We then check: do gpt-5.5's per-dim scores correlate with the
existing panel's A/B verdicts? If yes, the panel is measuring
something real. If no, the panel is judging style.

Scope: 30 (pid, base, ext) triples for W078 (the strongest KEEP).
~30 calls × 1 = 30 expensive calls (gpt-5.5 is one of the
'expensive' models available). Cost: ~$5-10.
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

# Add gpt-5.5 dynamically (it's not in EXPENSIVE_MODELS by default)
import os
from openai import OpenAI


class _GPT55Client:
    """Thin wrapper to call gpt-5.5 via the same proxy as the others."""
    def __init__(self):
        self.model = "gpt-5.5"
        self.family = "gpt55"
        self.provider = "newapi-gpt5/https://ruoli.dev/v1"
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


CACHE = PROJECT / "phase two" / "analysis" / "cache"
ANS = CACHE / "answers"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp38_gpt55_structured_audit_log.json"

PARALLEL = 4
N_PIDS = 30


RATING_PROMPT = """You are a careful methodological reviewer evaluating two
answers to the SAME problem. The two answers were produced by the same
solver model under two different scaffolding configurations. You do
NOT know which configuration produced which answer; do not speculate.

## Problem
{problem}

## Answer A
{answer_a}

## Answer B
{answer_b}

For each answer, give an independent 1-5 Likert score on FOUR
dimensions. Do not let the two answers' scores constrain each other.
Reason briefly before scoring (under 60 words). Then output JSON.

Dimensions (definitions):

1. **substantive_content** (1-5): Does the answer's actual technical
   claims address the problem? 5 = directly addresses the core, 1 =
   evades or substitutes generic talking points. Length is irrelevant;
   what matters is whether the claims engage the specific problem.

2. **methodological_soundness** (1-5): Is the reasoning structure
   valid? 5 = each conclusion follows from cited premises, 1 = leaps
   without justification or contradicts itself.

3. **problem_fit** (1-5): Does the answer engage what was actually
   asked? 5 = answers exactly the question posed, 1 = answers a
   different question.

4. **prescriptive_burden** (1-5, REVERSED): Does the answer
   over-prescribe / over-decide / load the recipient with
   non-actionable lists? 5 = appropriately calibrated to evidence,
   1 = over-prescribes (long checklists, fake-precision percentages,
   over-strong recommendations).

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
    """Return per-dim scores for both answers, randomising A/B side."""
    rng = random.Random(hash(pid) % (2**32))
    if rng.random() < 0.5:
        a, b, ext_was = ext, base, "A"
    else:
        a, b, ext_was = base, ext, "B"
    try:
        r = client.generate(RATING_PROMPT.format(problem=problem,
                                                  answer_a=a, answer_b=b),
                            max_tokens=1500, temperature=0.0)
        v = parse_json_from_llm(r["text"])
        sa = v.get("scores_a", {})
        sb = v.get("scores_b", {})
        # Map to ext / base scores
        if ext_was == "A":
            ext_scores, base_scores = sa, sb
        else:
            ext_scores, base_scores = sb, sa
        return {"pid": pid, "ext_was": ext_was,
                "ext_scores": ext_scores, "base_scores": base_scores,
                "reasoning_ext": v.get("reasoning_a" if ext_was == "A"
                                          else "reasoning_b", "")[:300]}
    except Exception as e:
        return {"pid": pid, "ext_was": ext_was, "error": str(e)[:80]}


def main():
    base_ans = json.loads((ANS / "_exp10_v20_base_answers.json").read_text())
    ext_ans = json.loads((ANS / "_exp10_v20_ext_WCROSSL01_answers.json").read_text())
    pid_to_prob = load_problems()
    common = sorted(set(base_ans) & set(ext_ans) & set(pid_to_prob))
    print(f"Common pids: {len(common)}")

    rng = random.Random(42)
    sample = rng.sample(common, min(N_PIDS, len(common)))
    print(f"Sampled {len(sample)} for gpt-5.5 structured rating")

    client = _GPT55Client()
    print(f"Rater: {client.model}")
    print(f"Per-dim Likert 1-5 across {len(sample)} (pid, base, ext) pairs.\n")

    results = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(rate_pair, client, pid_to_prob[p],
                            base_ans[p], ext_ans[p], p): p for p in sample}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            n = len(results)
            if n % 5 == 0:
                print(f"  {n}/{len(sample)} ({time.time()-t0:.0f}s)")

    # Aggregate per-dim diffs
    dims = ["substantive", "method", "fit", "prescriptive"]
    valid = [r for r in results if "error" not in r and r.get("ext_scores")
             and r.get("base_scores")]
    print(f"\nValid ratings: {len(valid)}/{len(results)}")

    print(f"\n=== Per-dim mean scores ===")
    print(f"{'dim':14s} {'ext mean':>10s} {'base mean':>10s} "
          f"{'diff (ext-base)':>17s} {'n_ext>base':>12s}")
    for d in dims:
        ext_vals = [r["ext_scores"].get(d) for r in valid
                     if r["ext_scores"].get(d) is not None]
        base_vals = [r["base_scores"].get(d) for r in valid
                      if r["base_scores"].get(d) is not None]
        # diffs (ext - base)
        diffs = []
        n_ext_better = 0
        for r in valid:
            es = r["ext_scores"].get(d)
            bs = r["base_scores"].get(d)
            if es is None or bs is None:
                continue
            diffs.append(es - bs)
            if es > bs: n_ext_better += 1
        if not ext_vals: continue
        em = sum(ext_vals) / len(ext_vals)
        bm = sum(base_vals) / len(base_vals)
        dm = sum(diffs) / len(diffs) if diffs else 0
        print(f"{d:14s} {em:>10.2f} {bm:>10.2f} {dm:>+17.2f} "
              f"{n_ext_better:>5d}/{len(diffs)}")

    # Combined headline: average across the 4 dims (treating prescriptive as REVERSED — higher = lower prescriptive burden = better)
    # Then compute "ext > base on combined score" rate
    n_ext_combined = 0; n_tie_combined = 0; n_base_combined = 0
    for r in valid:
        try:
            ext_avg = sum(r["ext_scores"][d] for d in dims) / 4
            base_avg = sum(r["base_scores"][d] for d in dims) / 4
            if ext_avg > base_avg + 0.1: n_ext_combined += 1
            elif base_avg > ext_avg + 0.1: n_base_combined += 1
            else: n_tie_combined += 1
        except: pass
    print(f"\nCombined (4-dim mean): "
          f"ext_better = {n_ext_combined}, base_better = {n_base_combined}, "
          f"tie = {n_tie_combined} (out of {len(valid)})")

    # Print 3 sample reasoning per direction
    samples_show = sorted(valid, key=lambda r: -(
        sum(r["ext_scores"].get(d, 3) for d in dims) -
        sum(r["base_scores"].get(d, 3) for d in dims)))
    print(f"\n=== Sample reasoning, top-3 EXT-leaning pids ===")
    for r in samples_show[:3]:
        print(f"  {r['pid']}: ext_avg = "
              f"{sum(r['ext_scores'].get(d, 0) for d in dims)/4:.2f}, "
              f"base_avg = {sum(r['base_scores'].get(d, 0) for d in dims)/4:.2f}")
        print(f"    {r['reasoning_ext'][:200]}")
    print(f"\n=== Sample reasoning, top-3 BASE-leaning pids ===")
    for r in samples_show[-3:]:
        print(f"  {r['pid']}: ext_avg = "
              f"{sum(r['ext_scores'].get(d, 0) for d in dims)/4:.2f}, "
              f"base_avg = {sum(r['base_scores'].get(d, 0) for d in dims)/4:.2f}")
        print(f"    {r['reasoning_ext'][:200]}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rater_model": "gpt-5.5",
           "rater_role": "structured Likert (1-5) on 4 dimensions",
           "n_pids_sampled": len(sample), "n_valid": len(valid),
           "candidate": "W078 (WCROSSL01)",
           "results": results}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
