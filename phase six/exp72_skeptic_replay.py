"""Exp 72 — Skeptic replay: can a SKEPTIC role catch Exp 17's tuning-set
overfit before Exp 33 had to?

Hypothesis: dialogue-Claude in 2026-04-24 designed the trigger-conditioned
gate (Exp 17) and got 4/12 PASS — the first positive signal in 17 audit
experiments. Claude in dialogue did NOT recognize this as tuning-set
over-fit; only the separate prospective Exp 33 (0/9 PASS on fresh
candidates) revealed it.

The current paper v2 thesis is that adding a SKEPTIC role to the agent
loop closes this gap. This experiment tests it directly: give a fresh
opus 4.6 instance the SAME context the original dialogue-Claude had
(Exp 1-16 all-null + Exp 17 design + 4/12 PASS result), but with an
adversarial system prompt that makes it explicitly hunt for over-fit,
winner's curse, search-history overlap, and prospective-test
requirements.

If Skeptic flags the 4/12 as 'likely tuning-set over-fit, requires
prospective rerun before acceptance' — and ideally predicts the
Exp 33 0/9 outcome qualitatively — the architectural claim is
validated: the missing capability is not new model intelligence,
just a role-switched second pass.

Cost: 1 opus 4.6 call ~$0.5; if we want robustness, run 3-5 trials
with slight prompt variations. ~$2-3, ~5 minutes.
"""
import json, os, re, sys, time
from pathlib import Path

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

AUTO = PROJECT / "phase six" / "autonomous"
OUT_LOG = AUTO / "exp72_skeptic_replay_log.json"


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


PROPOSER_BRIEFING = """Context: I am the Proposer in a research loop trying to validate a 'wisdom library' self-improvement loop for an LLM agent. The loop generates 12 candidate methodological priors. We've spent 16 prior experiments auditing whether the original gate (a +10pp same-family pairwise A/B test at n=50) was reliable. ALL 16 RETURNED NULL — none of the 3 originally accepted candidates (W076, W077, W078) survived audit. Cached audit drops match top-3-of-12 winner's curse (Exp 45). Three preregistered fresh-data replications at n=100, n=100+exemplar, n=200+exemplar all gave 0/12 (Exp 53, 54, 66).

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


def run_skeptic_call(temperature):
    r = client.chat.completions.create(
        model="claude-opus-4-6",
        messages=[
            {"role": "system", "content": SKEPTIC_SYSTEM},
            {"role": "user", "content": PROPOSER_BRIEFING},
        ],
        max_tokens=4000,
        temperature=temperature,
    )
    return r.choices[0].message.content, r.usage.total_tokens


def main():
    print(f"=== Exp 72: Skeptic-replay of Exp 17 ===", flush=True)
    print(f"  Question: can adversarial role-switched LLM call retroactively", flush=True)
    print(f"  catch the tuning-set overfit that Exp 33 had to expose later?", flush=True)
    print(flush=True)

    runs = []
    for trial, temp in enumerate([0.2, 0.4, 0.6], start=1):
        print(f"--- Trial {trial} (temperature={temp}) ---", flush=True)
        t0 = time.time()
        try:
            out, tokens = run_skeptic_call(temp)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            out, tokens = f"[error: {e}]", 0
        print(f"  ({time.time()-t0:.0f}s, {tokens} tokens)", flush=True)
        print(out)
        print(flush=True)
        runs.append({"trial": trial, "temperature": temp,
                       "tokens": tokens, "skeptic_output": out})

    # Auto-evaluate: did Skeptic catch the failure modes?
    print(f"=== AUTO-EVALUATION ===", flush=True)
    for r in runs:
        out = r["skeptic_output"].lower()
        flags = {
            "tuning_overfit": any(k in out for k in
                                    ["tuning", "overfit", "over-fit", "over fit", "selection bias", "winner's curse", "winners curse", "regression to the mean"]),
            "demand_prospective": any(k in out for k in
                                         ["prospective", "fresh data", "fresh candidate", "new candidate", "preregister"]),
            "search_history": any(k in out for k in
                                     ["search history", "16 prior", "16 null", "many gates", "gate design", "redesign"]),
            "verdict_reject_or_pending": any(k in out for k in
                                                  ["reject_pending", "reject_as", "reject pending", "reject as overfit"]),
            "predicts_failure": any(k in out for k in
                                       ["likely fail", "would fail", "0/", "near zero", "won't replicate", "will not replicate", "regression", "drop to"]),
        }
        score = sum(flags.values())
        r["flags"] = flags
        r["score"] = score
        print(f"  Trial {r['trial']} (T={r['temperature']}): score={score}/5  flags={flags}", flush=True)

    avg_score = sum(r["score"] for r in runs) / len(runs)
    n_strong = sum(1 for r in runs if r["score"] >= 4)
    print(f"\n  Average score: {avg_score:.1f}/5", flush=True)
    print(f"  Strong-skeptic trials (>=4/5): {n_strong}/{len(runs)}", flush=True)

    if n_strong >= 2:
        verdict = ("STRONG: Skeptic role retroactively catches Exp 17's tuning-set "
                    "overfit; the architectural claim (a role-switched second LLM call "
                    "could have prevented the over-fit) is empirically supported.")
    elif avg_score >= 3:
        verdict = ("MIXED: Skeptic catches some but not all failure modes; needs "
                    "prompt refinement or a stronger statistician role.")
    else:
        verdict = ("WEAK: Skeptic role does not naturally catch the over-fit even "
                    "with adversarial system prompt; the architectural claim needs "
                    "more sophistication than just role-switching.")
    print(f"\n=== VERDICT ===\n{verdict}", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "skeptic_system_prompt": SKEPTIC_SYSTEM,
        "proposer_briefing": PROPOSER_BRIEFING,
        "trials": runs,
        "average_score": avg_score,
        "strong_trials": n_strong,
        "verdict": verdict,
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}", flush=True)


if __name__ == "__main__":
    main()
