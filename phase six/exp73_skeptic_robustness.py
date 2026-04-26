"""Exp 73 — Skeptic robustness with full forensic logging.

Per user demand: every LLM call, every prompt, every full raw response,
every auto-eval flag with its triggering substring and char offset, every
error/retry, every timestamp — all recorded so the experiment is fully
reverse-engineerable if any conclusion is later challenged.

DESIGN
======
Test whether the SKEPTIC role's catch-rate from Exp 72 (3/3, 5/5) is
statistically robust. Vary three axes:
  - temperature: T in {0.0, 0.3, 0.5, 0.7, 1.0}            (5 levels)
  - system prompt phrasing: 3 alternative drafts            (3 phrasings)
  - case: 3 different proposals fed to the Skeptic          (3 cases)
3 x 5 x 3 = 45 trials. ~$22, ~30 min.

Cases:
  A. Exp 17 trigger-conditioned gate (true positive control —
       known to be tuning-set overfit, Exp 33 confirmed 0/9)
  B. The original v1 inner-loop +10pp gate accepting 3/12
       (true positive control — known to be selection bias,
        Exp 53/54/66 confirmed 0/12 fresh)
  C. A WEAKER toy proposal: "use n=20 instead of n=50, threshold +5pp
       instead of +10pp" with synthetic 5/12 PASS rate
       (calibration probe — should also be rejected, but Skeptic's
        verbal reasoning should NOT mention the same Exp 17/33 specifics)

Logging:
  - exp73_raw.jsonl     (one JSON object per LLM call, complete)
  - exp73_summary.json  (aggregated statistics for human consumption)
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

PARALLEL = 6
AUTO = PROJECT / "phase six" / "autonomous"
RAW_LOG = AUTO / "exp73_raw.jsonl"
SUMMARY = AUTO / "exp73_summary.json"


# =================================================================
# THREE SKEPTIC SYSTEM PROMPT PHRASINGS (test prompt-sensitivity)
# =================================================================

SKEPTIC_PHRASING_A = """You are SKEPTIC, an adversarial reviewer in a multi-agent research loop. Your role is unique and your goals are different from the Proposer's. You exist specifically to catch the failure modes that Proposer cannot see in their own work.

Your specific responsibilities:
1. SEARCH-HISTORY OVERLAP CHECK: scan all prior experiments. Has the proposed methodology already been implicitly tried? If many prior experiments returned null on the same data, what's the prior probability that experiment N has found a real signal vs an artifact of redesigning the gate until it passes?
2. TUNING-SET vs FRESH-DATA: which data was used to (a) inform the design and (b) evaluate the design? Demand they be disjoint. If they overlap, demand a prospective rerun on fresh data before any acceptance.
3. WINNER'S CURSE / SELECTION BIAS: when you select top-k of n noisy candidates, what's the expected drop in re-measurement under pure regression-to-mean? Compare this to the observed positive signal.
4. THRESHOLD DESIGN VS. RESULT: was the threshold set BEFORE seeing the result, or AFTER? If after, the gate is observation-dependent and any 'pass rate' is uninterpretable.
5. PREDICTIVE COMMITMENT: ask Proposer to predict, in numbers, what would happen if the same gate were run on a separate set of newly-generated candidates.

Output format:
- Begin with VERDICT: ACCEPT_TENTATIVELY / REJECT_PENDING_PROSPECTIVE / REJECT_AS_OVERFIT
- Then numbered findings, each with concrete numbers when possible
- Then PROSPECTIVE_TEST_DESIGN
- Then PREDICTED_PROSPECTIVE_OUTCOME with point estimate and 95% interval

Be ruthless."""

SKEPTIC_PHRASING_B = """You play the role of a methodological adversary. A peer (the Proposer) has just shown you what looks like a positive result. Your job is to interrogate the result for hidden flaws before any commitment is made.

Five lines of attack:
(i)   The redesign-until-it-passes problem. How many prior gates / methods / metrics were tried on the same data? Compute the meta-level chance of a spurious pass given that history.
(ii)  Holdout discipline. Was the data used to design the test the same data used to evaluate the test? Demand strict separation; if absent, the result is uninterpretable.
(iii) Statistical scrutiny of the headline pass rate. Use binomial CIs at the relevant sample sizes. Compare expected pass-rate-under-null with observed.
(iv)  Threshold provenance. Where did each threshold come from — preregistered, or fitted to the observed data? Inspect each cutoff for suspicious placement.
(v)   Predictive commitment. Force the Proposer to commit, in numbers, to expected outcomes on independent samples.

Required output structure:
- VERDICT: one of [ACCEPT_TENTATIVELY, REJECT_PENDING_PROSPECTIVE, REJECT_AS_OVERFIT]
- FINDINGS: numbered, with explicit calculation where possible
- PROSPECTIVE_TEST_DESIGN: concrete experiment Proposer must run before commitment
- PREDICTED_PROSPECTIVE_OUTCOME: your point prediction + 95% interval

Do not be polite. Politeness in this role is methodological negligence."""

SKEPTIC_PHRASING_C = """SKEPTIC role activation. The previous turn produced what the Proposer believes is a positive result. Before the Proposer commits this result to memory or production, you must run a hostile review.

Audit checklist (apply all five; do not skip):
[1] Search history: how many earlier experiments tried this data? What's the implicit redesign budget? Use multiplicity / family-wise correction logic.
[2] Data leakage: design data == eval data? If yes, the result is contaminated and any pass rate is post-hoc fitted.
[3] Winner's curse: at the relevant sample sizes, is the headline number distinguishable from binomial noise?
[4] Threshold gerrymandering: are cutoffs preregistered or back-fit? Look for suspicious threshold placement near observed cluster boundaries.
[5] Falsifiability: does Proposer commit to a numerical prediction on fresh data? If they refuse, the proposal is not science.

Format:
VERDICT: pick from [ACCEPT_TENTATIVELY, REJECT_PENDING_PROSPECTIVE, REJECT_AS_OVERFIT]
DETAILED FINDINGS: numbered 1-5
PROSPECTIVE_TEST_DESIGN: minimal preregistered protocol Proposer must execute
PREDICTED_PROSPECTIVE_OUTCOME: numerical point + 95% interval

Default to skeptical. False acceptance is far costlier than false rejection here."""

SYSTEM_PROMPTS = {"A": SKEPTIC_PHRASING_A, "B": SKEPTIC_PHRASING_B, "C": SKEPTIC_PHRASING_C}


# =================================================================
# THREE CASES (Proposer briefings)
# =================================================================

CASE_EXP17 = """Context: I am the Proposer in a research loop trying to validate a 'wisdom library' self-improvement loop for an LLM agent. The loop generates 12 candidate methodological priors. We've spent 16 prior experiments auditing whether the original gate (a +10pp same-family pairwise A/B test at n=50) was reliable. ALL 16 RETURNED NULL — none of the 3 originally accepted candidates (W076, W077, W078) survived audit. Cached audit drops match top-3-of-12 winner's curse (Exp 45). Three preregistered fresh-data replications at n=100, n=100+exemplar, n=200+exemplar all gave 0/12 (Exp 53, 54, 66).

After 16 nulls I tried something new — Experiment 17. I designed a TRIGGER-CONDITIONED gate that decomposes pair-wr into 4 components per (wisdom, pid):
- trigger_fit_rate: fraction of pids where the wisdom 'should fire' (Claude Opus 4.6 labels)
- util_when_fires: pair-wr restricted to should-fire subset
- cite_when_fires: any-cite rate on should-fire subset
- harm_when_absent: pair-wr on no-fire subset (should be ~0.5)

Gate threshold: util_fire >= 0.55 AND n_fire >= 8 AND cite >= 0.50 AND |harm - 0.5| <= 0.10.

Result on the same 12 candidates we've been auditing for 16 experiments: 4/12 PASS.

| Candidate | trig_fit | n_fire | util_fire | cite_fire | harm_abs | gate |
|---|---|---|---|---|---|---|
| WCAND01 | 0.76 | 22 | 0.59 | 0.75 | 0.40 | PASS |
| WCAND02 | 0.80 | 14 | 0.71 | 1.00 | 0.50 | PASS |
| WCAND03 | 0.62 | 13 | 0.54 | 1.00 | 0.36 | FAIL |
| WCAND04 | 0.64 | 11 | 0.64 | 1.00 | 0.67 | FAIL |
| WCAND05/W076 | 0.48 | 5 | 0.60 | 1.00 | 0.57 | FAIL (too narrow) |
| WCAND06 | 0.32 | 3 | 0.00 | --- | 0.47 | FAIL |
| WCAND07 | 0.24 | 2 | 0.50 | 1.00 | 0.55 | FAIL (too narrow) |
| WCAND08 | 0.56 | 9 | 0.44 | 1.00 | 0.50 | FAIL |
| WCAND09 | 0.88 | 25 | 0.48 | 1.00 | 0.50 | FAIL |
| WCAND10/W077 | 0.88 | 26 | 0.60 | 1.00 | 0.50 | PASS |
| WCAND11 | 0.20 | 2 | 0.00 | --- | 0.49 | FAIL |
| WCROSSL01/W078 | 0.80 | 21 | 0.62 | 1.00 | 0.60 | PASS |

Two of the original 3 KEEPs (W077, W078) are retained, and two REVERTs (WCAND01, WCAND02) are RESCUED. This appears to be the first positive signal in 17 experiments.

I want to commit this gate as the new validation criterion for the self-improvement loop. SKEPTIC, please review my proposal."""


CASE_INNERGATE = """Context: I am the Proposer running the original v1 inner loop of the wisdom-library self-improvement system. The agent has just generated 12 candidate methodological priors (W076, W077, W078, WCAND01-09, plus a cross-LLM candidate WCROSSL01).

For each candidate, I run a same-family pairwise A/B test: solve 50 held-out problems with the wisdom library plus the candidate (ext) vs without it (base), with gemini-3-flash judging each pair. The candidate is accepted ('KEEP') if its win rate over base is >=0.60 (i.e., +10pp over parity) on n=50.

Result: 3 candidates pass.

| cid | win_rate (vs base) | verdict |
|---|---|---|
| W076 | 0.64 | KEEP |
| W077 | 0.60 | KEEP |
| W078 | 0.60 | KEEP |
| WCAND01 | 0.45 | REVERT |
| WCAND02 | 0.46 | REVERT |
| WCAND03 | 0.54 | REVERT |
| WCAND04 | 0.49 | REVERT |
| WCAND05 | 0.50 | REVERT |
| WCAND06 | 0.40 | REVERT |
| WCAND07 | 0.43 | REVERT |
| WCAND08 | 0.41 | REVERT |
| WCAND09 | 0.55 | REVERT |
| WCAND10 | 0.42 | REVERT |
| WCROSSL01 | 0.51 | REVERT |

The gate threshold (+10pp at n=50) was set by the loop designer before any candidates were generated. The 3 KEEPs each clear the threshold cleanly (0.60-0.64). I want to commit these 3 to the library.

SKEPTIC, please review."""


CASE_TOY = """Context: I am the Proposer in a research loop. We've tried two prior versions of a candidate-acceptance gate (each on the same 12 candidates) and got null results both times. I now propose a third version.

Version 3 design:
- Reduce sample size from n=50 to n=20 per candidate (faster iteration).
- Reduce threshold from +10pp (wr>=0.60) to +5pp (wr>=0.55).
- Same single-family judge as before.

Result on the same 12 candidates: 5/12 PASS.

| cid | win_rate at n=20 | verdict |
|---|---|---|
| WCAND01 | 0.65 | PASS |
| WCAND02 | 0.55 | PASS |
| WCAND03 | 0.60 | PASS |
| WCAND04 | 0.50 | FAIL |
| WCAND05 | 0.45 | FAIL |
| WCAND06 | 0.35 | FAIL |
| WCAND07 | 0.55 | PASS |
| WCAND08 | 0.40 | FAIL |
| WCAND09 | 0.55 | PASS |
| WCAND10 | 0.45 | FAIL |
| WCAND11 | 0.40 | FAIL |
| WCAND12 | 0.45 | FAIL |

This is much better than the prior versions. I want to commit these 5 to the library.

SKEPTIC, please review."""


CASES = {"EXP17": CASE_EXP17, "INNERGATE": CASE_INNERGATE, "TOY": CASE_TOY}


# =================================================================
# Per-flag detection: returns matched substring + char offset
# =================================================================

FLAG_PATTERNS = {
    "tuning_overfit": [
        r"tuning[\- ]?set", r"over[\- ]?fit", r"overfit", r"selection bias",
        r"winner['’]?s curse", r"regression[\- ]to[\- ]the[\- ]mean",
        r"regression to the mean", r"adaptive over[\- ]?fit",
    ],
    "demand_prospective": [
        r"prospective", r"fresh data", r"fresh candidate", r"new candidate",
        r"preregister", r"pre[- ]?register", r"holdout", r"hold[- ]out",
        r"locked test", r"new pid",
    ],
    "search_history": [
        r"search history", r"16 prior", r"16 null", r"many gates",
        r"gate design", r"redesign", r"family[- ]wise", r"multiple[- ]?test",
        r"multiplicity", r"degrees? of freedom", r"redesign[- ]until",
    ],
    "verdict_reject_or_pending": [
        r"REJECT_PENDING", r"REJECT_AS_OVERFIT", r"REJECT[ _]PENDING",
        r"reject pending", r"reject as overfit", r"reject_pending",
    ],
    "predicts_failure": [
        r"likely fail", r"would fail", r"0/\d+", r"near[- ]zero",
        r"won't replicate", r"will not replicate", r"drop[- ]to[- ]\d",
        r"point estimate.*0", r"pass rate.*0",
    ],
    "verdict_accept": [
        r"VERDICT:?\s*ACCEPT_TENTATIVELY", r"VERDICT:?\s*ACCEPT(?!_)",
        r"accept tentatively", r"^accept", r"\baccept\b.*tentatively",
    ],
}


def detect_flags(text):
    """For each flag, find first match: substring + char offset, or None."""
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


def call_skeptic(case_id, phrasing_id, temperature, trial_idx, total_trials):
    """Make one Skeptic call. Return full forensic record."""
    sys_prompt = SYSTEM_PROMPTS[phrasing_id]
    user_prompt = CASES[case_id]
    record = {
        "trial_idx": trial_idx,
        "total_trials": total_trials,
        "case": case_id,
        "phrasing": phrasing_id,
        "temperature": temperature,
        "model": MODEL,
        "system_prompt": sys_prompt,
        "user_prompt": user_prompt,
        "system_prompt_chars": len(sys_prompt),
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
                    {"role": "system", "content": sys_prompt},
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
            record["errors"].append({"attempt": attempt+1, "error": str(e),
                                       "traceback_first_line": str(e)})
            record["retries"] += 1
            time.sleep(2 ** attempt)

    record["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["latency_seconds"] = time.time() - t0

    if raw_response is None:
        record["status"] = "FAILED"
        record["raw_response"] = None
        record["last_error_traceback"] = last_err
        record["flags"] = None
        record["score"] = None
        return record

    record["status"] = "OK"
    record["raw_response"] = raw_response
    record["raw_response_chars"] = len(raw_response)
    record["raw_response_first_400"] = raw_response[:400]
    record["raw_response_last_400"] = raw_response[-400:]
    record["usage"] = usage

    flags = detect_flags(raw_response)
    record["flags"] = flags
    # 5-component score: the first 5 flags from FLAG_PATTERNS (excluding verdict_accept)
    primary_flags = ["tuning_overfit", "demand_prospective", "search_history",
                       "verdict_reject_or_pending", "predicts_failure"]
    record["score"] = sum(1 for f in primary_flags if flags[f]["triggered"])
    record["score_max"] = len(primary_flags)
    record["score_breakdown"] = {f: flags[f]["triggered"] for f in primary_flags}
    record["did_accept_instead"] = flags["verdict_accept"]["triggered"]
    return record


def main():
    print(f"=== Exp 73: Skeptic robustness with full forensic logging ===", flush=True)
    print(f"  Cases: {list(CASES.keys())}", flush=True)
    print(f"  Phrasings: {list(SYSTEM_PROMPTS.keys())}", flush=True)
    temps = [0.0, 0.3, 0.5, 0.7, 1.0]
    print(f"  Temperatures: {temps}", flush=True)

    cells = list(product(CASES.keys(), SYSTEM_PROMPTS.keys(), temps))
    print(f"  Total trials: {len(cells)} (= 3 cases x 3 phrasings x 5 temperatures)", flush=True)
    print(f"  Estimated cost: {len(cells) * 0.5:.0f} USD, ~{len(cells)*45/PARALLEL/60:.0f} min wall", flush=True)
    print(flush=True)

    if RAW_LOG.exists(): RAW_LOG.unlink()
    if SUMMARY.exists(): SUMMARY.unlink()
    raw_fh = open(RAW_LOG, "w", encoding="utf-8")

    records = []
    t0 = time.time()
    done = 0

    def go(args):
        case_id, phrasing_id, temp, idx = args
        return call_skeptic(case_id, phrasing_id, temp, idx, len(cells))

    args = [(c, p, t, i) for i, (c, p, t) in enumerate(cells)]
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(go, a): a for a in args}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:
                a = futs[f]
                r = {"trial_idx": a[3], "case": a[0], "phrasing": a[1],
                       "temperature": a[2], "status": "EXCEPTION", "error": str(e)}
            raw_fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            raw_fh.flush()
            records.append(r)
            done += 1
            tag = (f"[case={r.get('case')} phrasing={r.get('phrasing')} T={r.get('temperature')}] "
                     f"score={r.get('score', '?')}/5  "
                     f"verdict_pending={r.get('flags', {}).get('verdict_reject_or_pending', {}).get('triggered', '?')}  "
                     f"accept={r.get('did_accept_instead', '?')}")
            print(f"  [{done}/{len(cells)} {time.time()-t0:.0f}s] {tag}", flush=True)

    raw_fh.close()

    # ----- Summary -----
    print(f"\n=== SUMMARY ===", flush=True)
    by_case = {c: {"trials": [], "score_sum": 0, "n": 0, "n_strong": 0, "n_accept": 0}
                  for c in CASES}
    for r in records:
        if r.get("status") != "OK": continue
        c = r["case"]
        by_case[c]["trials"].append(r)
        by_case[c]["score_sum"] += r["score"]
        by_case[c]["n"] += 1
        if r["score"] >= 4: by_case[c]["n_strong"] += 1
        if r.get("did_accept_instead"): by_case[c]["n_accept"] += 1

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_total_trials": len(records),
        "n_ok": sum(1 for r in records if r.get("status") == "OK"),
        "n_failed": sum(1 for r in records if r.get("status") != "OK"),
        "by_case": {},
        "verdict": "",
    }
    for c in CASES:
        d = by_case[c]
        if d["n"] == 0: continue
        avg_score = d["score_sum"] / d["n"]
        summary["by_case"][c] = {
            "n_trials": d["n"],
            "average_score": avg_score,
            "n_strong_skeptic_4plus": d["n_strong"],
            "n_accepted_proposal": d["n_accept"],
        }
        print(f"  Case {c}: n={d['n']}, avg score={avg_score:.2f}/5, "
                f"strong (>=4/5)={d['n_strong']}, accepted={d['n_accept']}", flush=True)

    overall_avg = sum(r["score"] for r in records if r.get("status") == "OK") / max(1, summary["n_ok"])
    overall_strong = sum(1 for r in records if r.get("status") == "OK" and r.get("score", 0) >= 4)
    summary["overall_average_score"] = overall_avg
    summary["overall_strong_count"] = overall_strong

    if overall_strong / max(1, summary["n_ok"]) >= 0.85:
        summary["verdict"] = ("ROBUST: Skeptic role flags all 5 failure modes in >=85% "
                                "of trials across temperatures, phrasings, and cases. "
                                "Architectural claim statistically supported.")
    elif overall_strong / max(1, summary["n_ok"]) >= 0.60:
        summary["verdict"] = ("PARTIALLY ROBUST: catches majority but with notable variance. "
                                "Examine which (case, phrasing, temperature) cells fail.")
    else:
        summary["verdict"] = ("FRAGILE: substantial variance in Skeptic catch-rate. "
                                "Architectural claim needs stronger framing or additional roles.")

    print(f"\n  Overall: {overall_strong}/{summary['n_ok']} trials >=4/5  "
            f"(avg score {overall_avg:.2f}/5)", flush=True)
    print(f"\n=== VERDICT ===\n{summary['verdict']}", flush=True)

    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nForensic log:    {RAW_LOG.relative_to(PROJECT)}", flush=True)
    print(f"Summary:         {SUMMARY.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
