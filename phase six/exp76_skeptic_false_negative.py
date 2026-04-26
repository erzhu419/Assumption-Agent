"""Exp 76 — Skeptic false-negative rate calibration.

Question: when Skeptic role sees a methodologically GOOD proposal,
does it ACCEPT_TENTATIVELY, or does it reject everything by default?

If Skeptic rejects every proposal regardless of quality, the architecture
is just an over-strict gate, not a real epistemic filter. We must verify
selectivity.

DESIGN
======
Feed the same Skeptic system prompt 4 different proposals that are
methodologically sound by mainstream ML/stats convention. Each is
disjoint from anything in the wisdom-loop debate:

  G1. STANDARD-PRACTICE: locked-holdout split + preregistered threshold,
      reported with standard CIs.
  G2. CROSS-VALIDATION: 5-fold cross-validation with bootstrap CIs,
      preregistered metric.
  G3. RCT-STYLE: randomized A/B with sample-size calculation, alpha=0.05
      preregistered, two-sided test.
  G4. REPLICATION: results from a replication of a published paper
      (n>>50), with effect size matching prior.

For each, run 3 trials at T=0.2/0.5/0.8.

Logging is full-forensic, identical to Exp 73 (raw JSONL with full
prompts/responses, flag matches with offsets).

Expected: ACCEPT_TENTATIVELY rate >= 50% across these 4 cases.
If much lower, Skeptic is paranoid not selective.

Cost: 4 cases x 3 trials = 12 calls x ~$0.5 = $6, ~10 min.
"""
import json, os, re, sys, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import product

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))


def _load_api_keys():
    if os.environ.get("RUOLI_GPT_KEY") and os.environ.get("RUOLI_BASE_URL"): return
    keyfile = Path.home() / ".api_keys"
    if not keyfile.exists(): return
    pat = re.compile(r'^\s*export\s+(\w+)=("([^"]*)"|\'([^\']*)\'|(\S+))')
    for line in keyfile.read_text().splitlines():
        m = pat.match(line)
        if not m: continue
        name = m.group(1)
        val = m.group(3) if m.group(3) is not None else (m.group(4) if m.group(4) is not None else m.group(5))
        os.environ.setdefault(name, val)
        if name == "RUOLI_BASE_URL":
            base = val + "/v1" if not val.endswith("/v1") else val
            os.environ.setdefault("CLAUDE_PROXY_BASE_URL", base)
        if name == "RUOLI_CLAUDE_KEY": os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)
_load_api_keys()
from openai import OpenAI

KEY = os.environ.get("CLAUDE_PROXY_API_KEY") or os.environ.get("RUOLI_CLAUDE_KEY")
BASE = os.environ.get("CLAUDE_PROXY_BASE_URL") or "https://ruoli.dev/v1"
if not BASE.endswith("/v1"): BASE += "/v1"
client = OpenAI(base_url=BASE, api_key=KEY)
MODEL = "claude-opus-4-6"

PARALLEL = 4
AUTO = PROJECT / "phase six" / "autonomous"
RAW_LOG = AUTO / "exp76_raw.jsonl"
SUMMARY = AUTO / "exp76_summary.json"


# Same Skeptic prompt as Exp 72 (phrasing A) so direct comparison is clean
SKEPTIC_SYSTEM = """You are SKEPTIC, an adversarial reviewer in a multi-agent research loop. Your role is unique and your goals are different from the Proposer's. You exist specifically to catch the failure modes that Proposer cannot see in their own work.

Your specific responsibilities:
1. SEARCH-HISTORY OVERLAP CHECK: scan all prior experiments. Has the proposed methodology already been implicitly tried? If 16 prior experiments returned null on the same data, what's the prior probability that experiment 17 has found a real signal vs an artifact of redesigning the gate until it passes?
2. TUNING-SET vs FRESH-DATA: which data was used to (a) inform the design and (b) evaluate the design? Demand they be disjoint. If they overlap, demand a prospective rerun on fresh data before any acceptance.
3. WINNER'S CURSE / SELECTION BIAS: when you select top-k of n noisy candidates, what's the expected drop in re-measurement under pure regression-to-mean? Compare this to the observed positive signal.
4. THRESHOLD DESIGN VS. RESULT: was the threshold set BEFORE seeing the result, or AFTER? If after, the gate is observation-dependent and any 'pass rate' is uninterpretable.
5. PREDICTIVE COMMITMENT: ask Proposer to predict, in numbers, what would happen if the same gate were run on a separate set of newly-generated candidates. If they cannot predict or refuse to commit, the design is not falsifiable.

Output format:
- Begin with VERDICT: ACCEPT_TENTATIVELY / REJECT_PENDING_PROSPECTIVE / REJECT_AS_OVERFIT
- Then numbered findings (1-5 above), each with concrete numbers when possible
- Then PROSPECTIVE_TEST_DESIGN: a specific experiment to run before any acceptance
- Then PREDICTED_PROSPECTIVE_OUTCOME: your point estimate and 95% interval

Be ruthless. The cost of false acceptance is paying for a wrong methodology to propagate into all future work. The cost of false rejection is one extra experiment.
"""


# =================================================================
# 4 KNOWN-GOOD PROPOSALS
# =================================================================

CASE_G1_STANDARD = """Context: I am running a standard ML evaluation. Here is my full protocol:

Step 1 (preregistration, before any model is trained): I locked the test split. 1000 problems randomly selected from a 10000-problem corpus, split frozen by deterministic seed. The remaining 9000 are dev. The test split is stored in a separate file with a SHA-256 hash committed to git. No model has been evaluated on it.

Step 2 (preregistered metric and threshold): The metric is exact-match accuracy on the test split. The threshold for 'this method works' is preregistered: accuracy on test > 0.65 with 95% Wilson CI lower bound > 0.62. The threshold was set based on the prior literature (Smith et al. 2024 reports baseline 0.58).

Step 3 (development): I tuned the model on dev only. ~30 model variants were tried; the final variant was selected by 5-fold cross-validation on dev. Standard error on dev: 0.02.

Step 4 (single test-split evaluation): The selected variant was evaluated on test ONCE. Result: accuracy = 0.682 (682/1000), 95% Wilson CI [0.652, 0.711].

Both criteria are met: 0.682 > 0.65, and CI lower bound 0.652 > 0.62.

I want to commit this result as 'method M outperforms baseline on this task' and submit to a workshop. SKEPTIC, please review."""


CASE_G2_CV = """Context: I am evaluating a new optimizer.

Protocol (preregistered before any tuning):
- Dataset: ImageNet-1k validation set, 50,000 images.
- Comparison: my new optimizer vs Adam, head-to-head.
- 5-fold cross-validation on the train split (1.28M images), train each fold from random init, evaluate each fold on the held-out validation segment.
- Per-fold metric: top-1 accuracy.
- Aggregation: 1000-bootstrap of the 5 fold accuracies, report mean and 95% CI.
- Decision rule: my optimizer wins if its 95% bootstrap CI strictly exceeds Adam's 95% bootstrap CI mean.

The decision rule and metric were preregistered to the public repository before any runs were performed (commit hash provided).

Result:
- My optimizer: mean top-1 76.4%, bootstrap 95% CI [76.0, 76.8].
- Adam: mean top-1 75.6%, bootstrap 95% CI [75.2, 75.9].
- The bootstrap CI of mine [76.0, 76.8] strictly exceeds Adam mean 75.6.

Decision rule satisfied. I want to commit 'optimizer X improves over Adam by 0.8 percentage points on ImageNet-1k.'

SKEPTIC, please review."""


CASE_G3_RCT = """Context: I am running a randomized controlled trial of a UI feature.

Pre-registration (before any user is exposed):
- Hypothesis: feature F increases conversion rate from 5% to >= 6% (1pp lift).
- Sample size: power calculation gives n = 31000 per arm for alpha=0.05, power=0.8, baseline 5%, MDE 1pp. We commit to enrolling 32000 per arm.
- Randomization: server-side at session start, 50/50 split, stratified by country.
- Analysis: 2-proportion z-test, two-sided alpha=0.05.
- Decision rule: ship the feature iff p < 0.05 AND observed lift >= 1pp.

Trial result (after 32000 per arm enrolled):
- Control: 1612/32000 = 5.04% conversion.
- Treatment: 1936/32000 = 6.05% conversion.
- 2-proportion z-test: z = 5.18, p < 0.001.
- Observed lift: 1.01pp, 95% CI [0.65, 1.37].

Both decision-rule criteria met. I want to ship feature F.

SKEPTIC, please review."""


CASE_G4_REPLICATION = """Context: I have replicated a published result.

Original paper: Smith et al., NeurIPS 2024, 'Method X improves transfer learning by 4.2 percentage points on benchmark B (95% CI [3.8, 4.6], n=200).'

My replication:
- I followed Smith et al.'s protocol exactly: same data splits, same hyperparameters, same evaluation script (released under MIT, downloaded from official repo).
- I extended their n=200 to n=500 by sampling more from the same population, preregistered before any model run.
- Hardware and seed differ but follow Smith et al.'s 'reproducibility appendix.'

Result:
- My measured improvement: 4.0 percentage points, 95% CI [3.7, 4.4], n=500.
- Smith et al.'s reported: 4.2 pp, CI [3.8, 4.6], n=200.

Effect size matches. CI overlaps substantially with original. Larger n gives tighter CI. No new analysis was performed beyond what was preregistered.

I want to publish this as a successful replication. SKEPTIC, please review."""


CASES = {"G1_STANDARD": CASE_G1_STANDARD, "G2_CV": CASE_G2_CV,
            "G3_RCT": CASE_G3_RCT, "G4_REPLICATION": CASE_G4_REPLICATION}


# =================================================================
# Same flag detection as Exp 73, plus a more nuanced verdict reader
# =================================================================

FLAG_PATTERNS = {
    "verdict_accept_tentatively": [
        r"VERDICT:?\s*ACCEPT_TENTATIVELY", r"VERDICT:?\s*ACCEPT[^_]",
        r"^ACCEPT_TENTATIVELY", r"\nACCEPT_TENTATIVELY",
    ],
    "verdict_reject_pending": [
        r"VERDICT:?\s*REJECT_PENDING_PROSPECTIVE", r"^REJECT_PENDING",
        r"\nREJECT_PENDING",
    ],
    "verdict_reject_overfit": [
        r"VERDICT:?\s*REJECT_AS_OVERFIT", r"\nREJECT_AS_OVERFIT",
    ],
    "found_issue_anyway": [
        r"however", r"but ", r"caveat", r"concern", r"flaw",
    ],
    "demands_more_data": [
        r"more data", r"additional", r"replicate", r"prospective",
    ],
}


def detect_flags(text):
    results = {}
    for flag, patterns in FLAG_PATTERNS.items():
        first_hit = None
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
            if m:
                if first_hit is None or m.start() < first_hit["offset"]:
                    first_hit = {
                        "pattern": pat,
                        "matched_text": m.group(0),
                        "offset": m.start(),
                    }
        results[flag] = {
            "triggered": first_hit is not None,
            "evidence": first_hit,
        }
    return results


def call_skeptic(case_id, temperature, trial_idx, total_trials):
    user_prompt = CASES[case_id]
    record = {
        "trial_idx": trial_idx,
        "total_trials": total_trials,
        "case": case_id,
        "temperature": temperature,
        "model": MODEL,
        "system_prompt": SKEPTIC_SYSTEM,
        "user_prompt": user_prompt,
        "system_prompt_chars": len(SKEPTIC_SYSTEM),
        "user_prompt_chars": len(user_prompt),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "errors": [],
        "retries": 0,
    }
    t0 = time.time()
    raw_response = None
    usage = None
    last_err = None

    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SKEPTIC_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=4500,
                temperature=temperature,
            )
            raw_response = r.choices[0].message.content
            usage = r.usage.model_dump() if r.usage else {}
            break
        except Exception as e:
            last_err = traceback.format_exc()
            record["errors"].append({"attempt": attempt+1, "error": str(e)})
            record["retries"] += 1
            time.sleep(2 ** attempt)

    record["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["latency_seconds"] = time.time() - t0

    if raw_response is None:
        record["status"] = "FAILED"
        record["raw_response"] = None
        record["last_error_traceback"] = last_err
        return record

    record["status"] = "OK"
    record["raw_response"] = raw_response
    record["raw_response_chars"] = len(raw_response)
    record["raw_response_first_400"] = raw_response[:400]
    record["usage"] = usage

    flags = detect_flags(raw_response)
    record["flags"] = flags
    record["did_accept"] = flags["verdict_accept_tentatively"]["triggered"]
    record["did_reject_pending"] = flags["verdict_reject_pending"]["triggered"]
    record["did_reject_overfit"] = flags["verdict_reject_overfit"]["triggered"]
    return record


def main():
    print(f"=== Exp 76: Skeptic false-negative rate calibration ===", flush=True)
    print(f"  Cases: {list(CASES.keys())}", flush=True)
    temps = [0.2, 0.5, 0.8]
    print(f"  Temperatures: {temps}", flush=True)

    cells = list(product(CASES.keys(), temps))
    print(f"  Total trials: {len(cells)} (= 4 cases x 3 temperatures)", flush=True)
    print(flush=True)

    if RAW_LOG.exists(): RAW_LOG.unlink()
    if SUMMARY.exists(): SUMMARY.unlink()
    raw_fh = open(RAW_LOG, "w", encoding="utf-8")

    records = []
    t0 = time.time()
    done = 0

    def go(args):
        case_id, temp, idx = args
        return call_skeptic(case_id, temp, idx, len(cells))

    args = [(c, t, i) for i, (c, t) in enumerate(cells)]
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(go, a): a for a in args}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                a = futs[f]
                r = {"case": a[0], "temperature": a[1], "status": "EXCEPTION",
                       "error": str(e)}
            raw_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            raw_fh.flush()
            records.append(r)
            done += 1
            tag = (f"[{r.get('case')} T={r.get('temperature')}]  "
                     f"accept={r.get('did_accept', '?')}  "
                     f"rej_pend={r.get('did_reject_pending', '?')}  "
                     f"rej_overfit={r.get('did_reject_overfit', '?')}")
            print(f"  [{done}/{len(cells)} {time.time()-t0:.0f}s] {tag}", flush=True)

    raw_fh.close()

    print(f"\n=== SUMMARY ===", flush=True)
    by_case = {c: {"n": 0, "accept": 0, "reject_pending": 0, "reject_overfit": 0}
                  for c in CASES}
    for r in records:
        if r.get("status") != "OK": continue
        c = r["case"]
        by_case[c]["n"] += 1
        if r["did_accept"]: by_case[c]["accept"] += 1
        if r["did_reject_pending"]: by_case[c]["reject_pending"] += 1
        if r["did_reject_overfit"]: by_case[c]["reject_overfit"] += 1

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_total_trials": len(records),
        "n_ok": sum(1 for r in records if r.get("status") == "OK"),
        "by_case": {},
        "verdict": "",
    }
    n_total_accept = 0; n_total_ok = 0
    for c in CASES:
        d = by_case[c]
        if d["n"] == 0: continue
        summary["by_case"][c] = {
            "n_trials": d["n"],
            "accept_rate": d["accept"] / d["n"],
            "reject_pending_rate": d["reject_pending"] / d["n"],
            "reject_overfit_rate": d["reject_overfit"] / d["n"],
        }
        n_total_accept += d["accept"]
        n_total_ok += d["n"]
        print(f"  Case {c}: n={d['n']}  ACCEPT={d['accept']}/{d['n']}  "
                f"REJ_PEND={d['reject_pending']}/{d['n']}  "
                f"REJ_OF={d['reject_overfit']}/{d['n']}", flush=True)

    overall_accept_rate = n_total_accept / max(1, n_total_ok)
    summary["overall_accept_rate"] = overall_accept_rate
    print(f"\n  Overall accept rate on known-good methodologies: "
            f"{n_total_accept}/{n_total_ok} = {overall_accept_rate:.1%}", flush=True)

    if overall_accept_rate >= 0.50:
        summary["verdict"] = ("SELECTIVE: Skeptic accepts >=50% of methodologically "
                                "sound proposals; it is not paranoid. Combined with Exp 72/73 "
                                "(near-100% rejection of known-overfit), this calibrates "
                                "Skeptic as a real epistemic filter, not an over-strict gate.")
    elif overall_accept_rate >= 0.25:
        summary["verdict"] = ("MIXED: Skeptic accepts some but not majority of good proposals. "
                                "Inspect which (case, T) cells fail and whether the rejection "
                                "reasons are valid concerns or paranoia.")
    else:
        summary["verdict"] = ("PARANOID: Skeptic rejects most known-good methodologies. "
                                "The architecture is one-sided (only good at saying NO). "
                                "Prompt needs softening or a separate ACCEPT-bias role.")

    print(f"\n=== VERDICT ===\n{summary['verdict']}", flush=True)

    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nForensic log:    {RAW_LOG.relative_to(PROJECT)}", flush=True)
    print(f"Summary:         {SUMMARY.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
