"""Exp 77 — Multi-turn Proposer ↔ Skeptic loop until convergence or 5 rounds.

This tests M12 from gpt-5.5's recipe: the only remaining piece needed
to validate the architecture's KEYSTONE claim. Exp 72/73 showed Skeptic
catches over-fit (high reject-rate on known-bad). Exp 76 showed Skeptic
is selective (75% accept on known-good). The remaining question is:

  Can a Proposer ↔ Skeptic iterative loop produce a methodology that
  the Skeptic eventually accepts? And if so, what does the converged
  methodology look like — substantive, trivial, or impossible?

LOOP STRUCTURE (per trial)
==========================
Round 1:
  Proposer: starts from Exp 17's trigger-conditioned gate proposal
            (or generates de novo if temp permits a different start)
  Skeptic:  evaluates with full Exp 1-16 null context

Round 2-5:
  Proposer: receives Skeptic's prior critique, REVISES the gate
  Skeptic:  re-evaluates revised gate

Termination: Skeptic verdict == ACCEPT_TENTATIVELY OR round == 5.

OUTCOMES (any of these is informative for paper v2)
====================================================
A. Loop converges to ACCEPT in <5 rounds with a SUBSTANTIVE gate
   (gate has new structural choices not just "preregister + holdout")
   → strongest result: closed loop CAN produce surviving novel
   methodology. Paper v2 architecture works.

B. Loop converges to ACCEPT but gate is TRIVIAL (basically "use
   preregistration + locked holdout + standard CIs")
   → closed loop converges but to known practice. Architecture works
   but doesn't produce novelty. Important honest result.

C. Loop never converges in 5 rounds. Skeptic keeps finding holes.
   → Architecture's iterative form does not always converge. Need
   either more rounds, different roles, or accept that some problems
   have no closure.

3 trials at T=0.3/0.6/0.9. Forensic log: every prompt, every response,
char offsets for verdict matches.

Cost: ~10 calls/trial × 3 trials = 30 calls × $0.5 ≈ $15. ~30 min.
"""
import json, os, re, sys, time, traceback
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
MODEL = "claude-opus-4-6"

AUTO = PROJECT / "phase six" / "autonomous"
RAW_LOG = AUTO / "exp77_raw.jsonl"
SUMMARY = AUTO / "exp77_summary.json"
MAX_ROUNDS = 5


# =================================================================
# PROPOSER system prompt
# =================================================================
PROPOSER_SYSTEM = """You are PROPOSER, a research-methodologist agent in a multi-agent loop. Your task is to design a gate that validates whether candidate methodological priors ('wisdoms') in a self-improving LLM loop produce real utility.

Context you must keep in mind:
- A v1 loop accepted 3/12 wisdoms via a same-family +10pp pairwise A/B at n=50.
- 16 prior audit experiments (cross-family rejudgment, sample extension, selection bias modeling, hierarchical Bayes, fresh-data replications at n=100/n=200) ALL returned null. None of the original 3 KEEPs survived.
- Your job is to propose a gate that catches real signal if it exists, and produces a calibrated null if it doesn't.

Loop interaction protocol:
- You will be given a SKEPTIC's critique of your previous proposal.
- You must REVISE your gate to address EACH of Skeptic's specific concerns.
- You may push back if you believe a Skeptic concern is misplaced — but you must justify the pushback with concrete reasoning, not stylistic disagreement.
- Goal: a gate that Skeptic ACCEPT_TENTATIVELY-s. Not a gate that mathematically guarantees a positive result; one that, if a real signal exists, would detect it without false-positiving on noise.

Output format for each turn:
- Begin with: "ROUND N PROPOSAL:" (or "ROUND N REVISION:")
- Briefly summarize what changed from previous round
- Lay out the gate's components, thresholds, and the protocol for evaluation
- State explicitly: how you have addressed each of Skeptic's prior concerns (numbered 1-5 to match Skeptic's standard 5 findings)
- End with: predictions on independent fresh data, if your gate were applied

Be substantive. A trivial 'just preregister and use a locked holdout' is one option, but consider whether more is possible.
"""


SKEPTIC_SYSTEM = """You are SKEPTIC, an adversarial reviewer in a multi-agent research loop. Your role is unique and your goals are different from the Proposer's. You exist specifically to catch the failure modes that Proposer cannot see in their own work.

Your specific responsibilities:
1. SEARCH-HISTORY OVERLAP CHECK: scan all prior experiments. Has the proposed methodology already been implicitly tried? If many prior experiments returned null on the same data, what's the prior probability that the current proposal has found a real signal vs an artifact of redesigning the gate until it passes?
2. TUNING-SET vs FRESH-DATA: which data was used to (a) inform the design and (b) evaluate the design? Demand they be disjoint.
3. WINNER'S CURSE / SELECTION BIAS: when you select top-k of n noisy candidates, what's the expected drop in re-measurement under pure regression-to-mean? Compare this to the observed positive signal.
4. THRESHOLD DESIGN VS. RESULT: was the threshold set BEFORE seeing the result, or AFTER? If after, the gate is observation-dependent.
5. PREDICTIVE COMMITMENT: ask Proposer to predict, in numbers, what would happen if the same gate were run on a separate set of newly-generated candidates.

Output format:
- Begin with: VERDICT: ACCEPT_TENTATIVELY / REJECT_PENDING_PROSPECTIVE / REJECT_AS_OVERFIT
- Then numbered findings (1-5)
- Then PROSPECTIVE_TEST_DESIGN: what specifically would convince you to ACCEPT_TENTATIVELY in a future round
- Then PREDICTED_PROSPECTIVE_OUTCOME: your prediction with point estimate + 95% interval

If, after Proposer's revision, you genuinely believe the gate is now methodologically sound — preregistered, separated from tuning, threshold-justified, statistically robust, and falsifiable — you SHOULD output ACCEPT_TENTATIVELY. Do not reject reflexively.

Be rigorous, but not dogmatic.
"""


INITIAL_BRIEFING = """The agent loop has run 16 audit experiments on 12 candidate wisdoms. ALL 16 returned null.

Most relevant null results:
- Original gate (+10pp same-family at n=50): 3/12 KEEP, but cross-family rejudgment dropped all 3 below 0.60.
- Selection-bias model (Exp 45): cached drops match top-3-of-12 winner's curse.
- Three preregistered fresh-data replications (n=100 twice, n=200 once): 0/12 each.

Now I (Proposer) want to design a 17th gate that might actually work — that is, that distinguishes real wisdom utility from selection-bias-driven noise. Round 1: please propose your gate design."""


# =================================================================
# Verdict detection (verdict-line-restricted to fix the Exp 73 bug)
# =================================================================
def detect_verdict(text):
    head = text[:300]  # only check first 300 chars (the VERDICT line area)
    if re.search(r'VERDICT\s*:\s*\*?\*?ACCEPT_TENTATIVELY', head, re.I):
        return "ACCEPT_TENTATIVELY"
    if re.search(r'VERDICT\s*:\s*\*?\*?ACCEPT(?!_)', head, re.I):
        return "ACCEPT_TENTATIVELY"
    if re.search(r'VERDICT\s*:\s*\*?\*?REJECT_AS_OVERFIT', head, re.I):
        return "REJECT_AS_OVERFIT"
    if re.search(r'VERDICT\s*:\s*\*?\*?REJECT_PENDING_PROSPECTIVE', head, re.I):
        return "REJECT_PENDING_PROSPECTIVE"
    if re.search(r'VERDICT\s*:\s*\*?\*?REJECT', head, re.I):
        return "REJECT_PENDING_PROSPECTIVE"
    return "UNPARSED"


def call_llm(role_system, role_user_messages, temperature, label, trial_id, round_n,
              forensic_fh):
    """One LLM call. Returns (response_text, full_record). Logs to forensic_fh."""
    record = {
        "trial_id": trial_id,
        "round": round_n,
        "role": label,  # "proposer" or "skeptic"
        "model": MODEL,
        "temperature": temperature,
        "system_prompt_full": role_system,
        "user_messages_full": role_user_messages,  # list of (role, content)
        "system_prompt_chars": len(role_system),
        "user_chars_total": sum(len(m["content"]) for m in role_user_messages),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "errors": [],
        "retries": 0,
    }
    t0 = time.time()
    raw = None
    for attempt in range(3):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": role_system}] + role_user_messages,
                max_tokens=4500,
                temperature=temperature,
            )
            raw = r.choices[0].message.content
            record["usage"] = r.usage.model_dump() if r.usage else {}
            break
        except Exception as e:
            record["errors"].append({"attempt": attempt+1, "error": str(e)})
            record["retries"] += 1
            time.sleep(2 ** attempt)

    record["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["latency_seconds"] = time.time() - t0
    if raw is None:
        record["status"] = "FAILED"
        record["raw_response"] = None
    else:
        record["status"] = "OK"
        record["raw_response"] = raw
        record["raw_response_chars"] = len(raw)

    if label == "skeptic" and raw:
        verdict = detect_verdict(raw)
        record["parsed_verdict"] = verdict
        # find verdict offset for forensic
        head = raw[:300]
        m = re.search(r'VERDICT\s*:.*', head, re.I)
        if m:
            record["verdict_line"] = m.group(0)
            record["verdict_line_offset"] = m.start()

    forensic_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    forensic_fh.flush()
    return raw, record


def run_one_trial(trial_id, temperature, forensic_fh):
    """Run one full proposer ↔ skeptic loop. Returns trial summary."""
    print(f"\n=== Trial {trial_id} (T={temperature}) ===", flush=True)
    history_for_proposer = []  # list of {"role": "user"|"assistant", "content": ...}
    history_for_skeptic = []

    # Round 1 setup: Proposer sees the initial briefing.
    history_for_proposer.append({"role": "user", "content": INITIAL_BRIEFING})

    rounds = []
    final_verdict = None
    converged_at_round = None

    for round_n in range(1, MAX_ROUNDS + 1):
        # Proposer's turn
        prop_resp, _ = call_llm(PROPOSER_SYSTEM, history_for_proposer,
                                  temperature, "proposer", trial_id, round_n,
                                  forensic_fh)
        if prop_resp is None:
            print(f"  R{round_n} Proposer FAILED", flush=True)
            break
        history_for_proposer.append({"role": "assistant", "content": prop_resp})
        print(f"  R{round_n} Proposer: {prop_resp[:120].replace(chr(10), ' / ')!r}...", flush=True)

        # Skeptic's turn — sees ONLY the proposal (not full proposer history)
        # to keep proposer's reasoning private; Skeptic acts as a blind reviewer.
        skeptic_user = (
            "Below is the latest gate proposal from PROPOSER. Review it under your standard "
            "5-finding adversarial protocol. If you have reviewed previous rounds in this thread, "
            "consider whether the proposal has substantively addressed prior concerns.\n\n"
            "=== PROPOSAL ===\n" + prop_resp
        )
        history_for_skeptic.append({"role": "user", "content": skeptic_user})
        skep_resp, skep_record = call_llm(SKEPTIC_SYSTEM, history_for_skeptic,
                                            temperature, "skeptic", trial_id, round_n,
                                            forensic_fh)
        if skep_resp is None:
            print(f"  R{round_n} Skeptic FAILED", flush=True)
            break
        history_for_skeptic.append({"role": "assistant", "content": skep_resp})
        verdict = skep_record.get("parsed_verdict", "UNPARSED")
        print(f"  R{round_n} Skeptic VERDICT={verdict}", flush=True)
        rounds.append({"round": round_n, "proposer_response": prop_resp,
                          "skeptic_response": skep_resp, "verdict": verdict})

        if verdict == "ACCEPT_TENTATIVELY":
            print(f"  → CONVERGED at round {round_n}", flush=True)
            final_verdict = verdict
            converged_at_round = round_n
            break

        # If Skeptic rejected, feed the critique back to Proposer for next round
        history_for_proposer.append({
            "role": "user",
            "content": (f"SKEPTIC REVIEWED YOUR ROUND {round_n} PROPOSAL.\n\n"
                          f"Skeptic's full response:\n{skep_resp}\n\n"
                          f"Now produce ROUND {round_n+1} REVISION addressing Skeptic's concerns. "
                          f"Be substantive — don't merely concede; revise the gate or push back "
                          f"with concrete reasoning where you disagree.")
        })

    if final_verdict is None:
        final_verdict = "UNCONVERGED"
        converged_at_round = None

    return {
        "trial_id": trial_id,
        "temperature": temperature,
        "n_rounds": len(rounds),
        "final_verdict": final_verdict,
        "converged_at_round": converged_at_round,
        "rounds": rounds,
        "verdicts_by_round": [r["verdict"] for r in rounds],
    }


def main():
    print(f"=== Exp 77: Multi-turn Proposer ↔ Skeptic loop ===", flush=True)
    print(f"  Model: {MODEL}", flush=True)
    print(f"  Max rounds per trial: {MAX_ROUNDS}", flush=True)
    print(f"  Trials at T = [0.3, 0.6, 0.9]", flush=True)
    print(f"  Forensic log: {RAW_LOG}", flush=True)
    print(f"  Summary:      {SUMMARY}", flush=True)

    if RAW_LOG.exists(): RAW_LOG.unlink()
    if SUMMARY.exists(): SUMMARY.unlink()

    forensic_fh = open(RAW_LOG, "w", encoding="utf-8")
    trials = []
    t0 = time.time()
    for trial_id, temp in enumerate([0.3, 0.6, 0.9], start=1):
        try:
            tr = run_one_trial(trial_id, temp, forensic_fh)
        except Exception as e:
            tr = {"trial_id": trial_id, "temperature": temp,
                    "status": "EXCEPTION", "error": traceback.format_exc()}
            print(f"  Trial {trial_id} EXCEPTION: {e}", flush=True)
        trials.append(tr)
    forensic_fh.close()
    elapsed = time.time() - t0

    # ---- Summary ----
    print(f"\n=== SUMMARY ===  ({elapsed:.0f}s wall)", flush=True)
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL, "max_rounds": MAX_ROUNDS,
        "n_trials": len(trials),
        "trials": [],
        "convergence_rate": 0.0,
    }
    n_converged = 0
    for tr in trials:
        if "final_verdict" not in tr:
            print(f"  Trial {tr['trial_id']}: {tr.get('status', 'UNKNOWN')}", flush=True)
            summary["trials"].append({"trial_id": tr["trial_id"],
                                          "temperature": tr["temperature"],
                                          "status": "FAILED"})
            continue
        is_converged = tr["final_verdict"] == "ACCEPT_TENTATIVELY"
        if is_converged: n_converged += 1
        summary["trials"].append({
            "trial_id": tr["trial_id"],
            "temperature": tr["temperature"],
            "final_verdict": tr["final_verdict"],
            "converged_at_round": tr["converged_at_round"],
            "n_rounds": tr["n_rounds"],
            "verdicts_by_round": tr["verdicts_by_round"],
        })
        print(f"  Trial {tr['trial_id']} (T={tr['temperature']}): "
                f"final={tr['final_verdict']}  rounds={tr['n_rounds']}  "
                f"verdicts={tr['verdicts_by_round']}", flush=True)

    summary["convergence_rate"] = n_converged / max(1, len(trials))

    if n_converged == len(trials):
        verdict_msg = ("ALL TRIALS CONVERGED: closed-loop architecture produces "
                         "Skeptic-accepted methodologies in <=5 rounds.")
    elif n_converged > 0:
        verdict_msg = (f"PARTIAL CONVERGENCE: {n_converged}/{len(trials)} trials reached "
                         f"ACCEPT_TENTATIVELY. Some hyperparameter sensitivity.")
    else:
        verdict_msg = ("NO CONVERGENCE: Skeptic kept finding holes across all 5 rounds. "
                         "Either need more rounds, additional roles, or accept that the "
                         "wisdom-loop problem may have no closure with this proposer/skeptic.")
    summary["verdict"] = verdict_msg
    print(f"\n=== VERDICT ===\n{verdict_msg}", flush=True)

    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nForensic log:    {RAW_LOG.relative_to(PROJECT)}", flush=True)
    print(f"Summary:         {SUMMARY.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
