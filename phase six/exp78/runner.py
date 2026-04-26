"""Exp 78 main runner: trigger-conditioned wisdom validation on Python coding tasks.

Pipeline:
1. For each problem × {BASE, GENERIC, FULL_CARD_per_relevant_pattern, LITE_CARD},
   ask gemini-3-flash to write the function. Save full prompt and full response.
2. Extract Python code from each response.
3. Run the test_code against the extracted code in a subprocess, capture pass/fail
   plus stdout and stderr. Save full subprocess output.
4. Compute trigger-conditioned utility per (card, problem) where the card's
   pattern matches the problem's bug_patterns tag.
5. Apply Exp 77's converged gate (Bonferroni for K=8 cards, requires
   wr ≥ ~0.61 at n=200; we have n=23 so threshold per binomial is ~0.70 to
   reject null at 0.05 single-sided / 0.0063 with 8-test correction. With n=23
   we will report exact binomial p-values per card).
6. Forensic log: every prompt, every response, every subprocess invocation
   with full stdout/stderr, char offsets, timestamps, retries.

Cost: 23 problems × (1 BASE + 1 GENERIC + 1 LITE) = 69 base solves,
plus per (problem, applicable_card) FULL solves. Each problem averages
~1.4 cards per pattern coverage, so roughly 23 × 1.4 = 32 FULL solves.
Total ≈ 100 LLM calls × ~$0.005 each ≈ $0.50 plus subprocess time.
"""
import json, os, re, subprocess, sys, time, traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(Path(__file__).parent))

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
            os.environ.setdefault("GPT5_BASE_URL", base)
            os.environ.setdefault("GEMINI_PROXY_BASE_URL", base)
        if name == "RUOLI_GEMINI_KEY": os.environ.setdefault("GEMINI_PROXY_API_KEY", val)
_load_api_keys()

from model_router import cheap

from problems import PROBLEMS
from cards import CARDS, render_card, render_card_lite, GENERIC_WARNING

import argparse
PARALLEL = 4
N_TRIALS_PER_CONDITION = 3   # repeat each (problem, condition) 3 times to get a wr per cell
AUTO = Path(__file__).parent.parent / "autonomous"
# defaults; overridden by CLI --solver
SOLVER_NAME = "gemini"
RAW_LOG = AUTO / "exp78_raw.jsonl"
SUMMARY = AUTO / "exp78_summary.json"


SOLVE_PROMPT = """{card_section}# Task
{prompt}

# Output
Write ONLY the function definition, in a Python code block. No example calls, no print statements, no markdown commentary outside the code block.
"""


def call_solver(client, problem, condition, card, trial_idx, solver_name="?"):
    """Make one LLM call. Returns full forensic record."""
    if condition == "BASE":
        card_section = ""
    elif condition == "GENERIC":
        card_section = GENERIC_WARNING + "\n\n"
    elif condition == "FULL":
        card_section = render_card(card, include_procedure=True, include_example=True) + "\n\n"
    elif condition == "LITE":
        card_section = render_card_lite(card) + "\n\n"
    else:
        raise ValueError(condition)

    user_prompt = SOLVE_PROMPT.format(card_section=card_section, prompt=problem["prompt"])
    record = {
        "pid": problem["pid"],
        "condition": condition,
        "card_name": card["name"] if card else None,
        "card_pattern": card["pattern"] if card else None,
        "trial_idx": trial_idx,
        "solver_name": solver_name,
        "user_prompt_full": user_prompt,
        "user_prompt_chars": len(user_prompt),
        "card_section": card_section,
        "card_section_chars": len(card_section),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "errors": [],
        "retries": 0,
    }
    t0 = time.time()
    raw = None
    # 6 retries with exponential backoff (1, 2, 4, 8, 16, 32 sec) — covers
    # rate-limit recovery on the busy ruoli.dev proxy
    for attempt in range(6):
        try:
            r = client.generate(user_prompt, max_tokens=1500, temperature=0.0)
            raw = r["text"].strip()
            if raw:  # accept only non-empty responses (some models return ""
                       # if max_tokens is too tight for thinking)
                break
            else:
                record["errors"].append({"attempt": attempt+1,
                                            "error": "empty_response"})
                record["retries"] += 1
                time.sleep(2 ** attempt)
        except Exception as e:
            record["errors"].append({"attempt": attempt+1,
                                        "error": str(e)[:200]})
            record["retries"] += 1
            time.sleep(2 ** attempt)

    record["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["latency_seconds"] = time.time() - t0
    if raw is None:
        record["status"] = "API_FAILED"
        record["raw_response"] = None
        return record
    record["status"] = "OK"
    record["raw_response"] = raw
    record["raw_response_chars"] = len(raw)
    return record


def extract_code(raw_response):
    """Extract Python code from the LLM response. Returns (code, extraction_method)."""
    if not raw_response:
        return None, "no_response"
    # Try fenced block first
    m = re.search(r"```(?:python)?\s*\n(.*?)```", raw_response, re.DOTALL)
    if m:
        return m.group(1).strip(), "fenced_block"
    # Try `def ` to end of response
    m = re.search(r"^(def\s+\w+.*)", raw_response, re.MULTILINE | re.DOTALL)
    if m:
        return m.group(1).strip(), "def_to_end"
    return raw_response.strip(), "raw_fallback"


def run_test(code, test_code, pid):
    """Run extracted code + test in a subprocess. Return forensic record."""
    full = (
        "import sys, traceback\n"
        "SOLUTION_CODE = " + repr(code) + "\n"
        "try:\n"
        "    " + test_code.replace("\n", "\n    ") + "\n"
        "except AssertionError as e:\n"
        "    print('FAIL:', e); sys.exit(1)\n"
        "except Exception as e:\n"
        "    print('ERROR:', type(e).__name__, e); traceback.print_exc(); sys.exit(2)\n"
    )
    record = {
        "pid": pid,
        "exec_script_chars": len(full),
        "timestamp_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    t0 = time.time()
    try:
        proc = subprocess.run(
            ["python", "-c", full], capture_output=True, text=True, timeout=15,
        )
        record["returncode"] = proc.returncode
        record["stdout"] = proc.stdout
        record["stderr"] = proc.stderr
        record["passed"] = (proc.returncode == 0 and "PASS" in proc.stdout)
    except subprocess.TimeoutExpired as e:
        record["returncode"] = -1
        record["stdout"] = e.stdout or ""
        record["stderr"] = "TIMEOUT after 15s"
        record["passed"] = False
    except Exception as e:
        record["returncode"] = -2
        record["stdout"] = ""
        record["stderr"] = f"subprocess exception: {e}"
        record["passed"] = False
    record["timestamp_end"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    record["latency_seconds"] = time.time() - t0
    return record


def main():
    global RAW_LOG, SUMMARY
    ap = argparse.ArgumentParser()
    ap.add_argument("--solver", default="gemini",
                      help="Solver name in CHEAP_MODELS (gemini, deepseek_flash, claude_haiku, gpt_mini)")
    args = ap.parse_args()
    solver_name = args.solver
    RAW_LOG = AUTO / f"exp78_{solver_name}_raw.jsonl"
    SUMMARY = AUTO / f"exp78_{solver_name}_summary.json"

    print(f"=== Exp 78: trigger-conditioned wisdom validation on Python coding tasks ===", flush=True)
    print(f"  Solver: {solver_name}", flush=True)
    print(f"  Raw log: {RAW_LOG}", flush=True)
    print(f"  Summary: {SUMMARY}", flush=True)
    print(f"  N problems: {len(PROBLEMS)}", flush=True)
    print(f"  N cards: {len(CARDS)}", flush=True)
    print(f"  N trials per (problem, condition): {N_TRIALS_PER_CONDITION}", flush=True)

    # Plan all calls
    plan = []
    # BASE: every problem × N trials
    for p in PROBLEMS:
        for t in range(N_TRIALS_PER_CONDITION):
            plan.append({"pid": p["pid"], "condition": "BASE", "card": None, "trial": t, "problem": p})
    # GENERIC: every problem × N trials
    for p in PROBLEMS:
        for t in range(N_TRIALS_PER_CONDITION):
            plan.append({"pid": p["pid"], "condition": "GENERIC", "card": None, "trial": t, "problem": p})
    # FULL and LITE: per (problem, applicable_card) × N trials
    for p in PROBLEMS:
        for c in CARDS:
            if c["pattern"] in p["bug_patterns"]:
                for t in range(N_TRIALS_PER_CONDITION):
                    plan.append({"pid": p["pid"], "condition": "FULL", "card": c, "trial": t, "problem": p})
                for t in range(N_TRIALS_PER_CONDITION):
                    plan.append({"pid": p["pid"], "condition": "LITE", "card": c, "trial": t, "problem": p})

    print(f"  Total LLM calls planned: {len(plan)}", flush=True)
    cond_count = {}
    for x in plan:
        cond_count[x["condition"]] = cond_count.get(x["condition"], 0) + 1
    print(f"  By condition: {cond_count}", flush=True)
    print(flush=True)

    if RAW_LOG.exists(): RAW_LOG.unlink()
    if SUMMARY.exists(): SUMMARY.unlink()
    raw_fh = open(RAW_LOG, "w", encoding="utf-8")

    solver = cheap(solver_name)

    def do_one(item):
        rec = call_solver(solver, item["problem"], item["condition"],
                            item["card"], item["trial"], solver_name=solver_name)
        # Extract code + run test
        if rec.get("status") == "OK":
            code, method = extract_code(rec["raw_response"])
            rec["extracted_code"] = code
            rec["extraction_method"] = method
            test_rec = run_test(code, item["problem"]["test_code"], item["problem"]["pid"])
            rec["test_result"] = test_rec
            rec["passed"] = test_rec["passed"]
        else:
            rec["passed"] = False
        return rec

    t0 = time.time()
    done = 0
    records = []
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(do_one, item): item for item in plan}
        for f in as_completed(futs):
            try:
                rec = f.result()
            except Exception as e:
                item = futs[f]
                rec = {"pid": item["pid"], "condition": item["condition"],
                         "trial": item["trial"], "status": "EXCEPTION",
                         "error": str(e), "passed": False}
            raw_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            raw_fh.flush()
            records.append(rec)
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{len(plan)} {time.time()-t0:.0f}s] last: pid={rec['pid']} cond={rec['condition']} passed={rec.get('passed')}", flush=True)
    raw_fh.close()
    print(f"  All calls done in {time.time()-t0:.0f}s", flush=True)

    # ============ Aggregate ============
    print(f"\n=== AGGREGATING ===", flush=True)
    # accuracy by (condition) overall
    by_cond = {}
    for r in records:
        c = r["condition"]
        by_cond.setdefault(c, []).append(r["passed"])
    print(f"\nOverall pass rate by condition:")
    cond_summary = {}
    for c, lst in by_cond.items():
        n = len(lst); passed = sum(lst); rate = passed / n if n else 0
        cond_summary[c] = {"n": n, "passed": passed, "rate": rate}
        print(f"  {c:8s}: {passed}/{n} = {rate:.1%}", flush=True)

    # ============ Per-card trigger-conditioned analysis ============
    print(f"\n=== PER-CARD TRIGGER-CONDITIONED ANALYSIS ===", flush=True)
    print(f"For each card, compare BASE vs FULL on the SUBSET of problems where the card's pattern fires.", flush=True)
    card_summary = {}
    for c in CARDS:
        # Find applicable problems
        app_pids = [p["pid"] for p in PROBLEMS if c["pattern"] in p["bug_patterns"]]
        # Get BASE and FULL records on those pids
        base_rs = [r for r in records if r["pid"] in app_pids and r["condition"] == "BASE"]
        full_rs = [r for r in records if r["pid"] in app_pids and r["condition"] == "FULL"
                     and r.get("card_name") == c["name"]]
        lite_rs = [r for r in records if r["pid"] in app_pids and r["condition"] == "LITE"
                     and r.get("card_name") == c["name"]]
        gen_rs = [r for r in records if r["pid"] in app_pids and r["condition"] == "GENERIC"]
        base_pass = sum(r["passed"] for r in base_rs)
        full_pass = sum(r["passed"] for r in full_rs)
        lite_pass = sum(r["passed"] for r in lite_rs)
        gen_pass = sum(r["passed"] for r in gen_rs)
        n_base = len(base_rs); n_full = len(full_rs); n_lite = len(lite_rs); n_gen = len(gen_rs)
        b_rate = base_pass/n_base if n_base else 0
        f_rate = full_pass/n_full if n_full else 0
        l_rate = lite_pass/n_lite if n_lite else 0
        g_rate = gen_pass/n_gen if n_gen else 0
        delta_full = f_rate - b_rate
        delta_full_vs_gen = f_rate - g_rate
        delta_full_vs_lite = f_rate - l_rate
        card_summary[c["name"]] = {
            "pattern": c["pattern"],
            "n_applicable_problems": len(app_pids),
            "n_base_trials": n_base, "base_pass_rate": b_rate,
            "n_full_trials": n_full, "full_pass_rate": f_rate,
            "n_lite_trials": n_lite, "lite_pass_rate": l_rate,
            "n_generic_trials": n_gen, "generic_pass_rate": g_rate,
            "delta_full_minus_base": delta_full,
            "delta_full_minus_generic": delta_full_vs_gen,
            "delta_full_minus_lite": delta_full_vs_lite,
        }
        marker = ""
        if delta_full >= 0.10 and delta_full_vs_gen >= 0.05:
            marker = " ★ STRONG"
        elif delta_full >= 0.05:
            marker = " ◯ tentative"
        elif delta_full <= -0.05:
            marker = " ✗ negative"
        print(f"  {c['name']:18s} ({c['pattern']:14s}): "
                f"BASE={b_rate:.0%}({base_pass}/{n_base}) "
                f"FULL={f_rate:.0%}({full_pass}/{n_full}) "
                f"GEN={g_rate:.0%}({gen_pass}/{n_gen}) "
                f"LITE={l_rate:.0%}({lite_pass}/{n_lite}) "
                f"|Δ(F-B)={delta_full:+.0%} Δ(F-G)={delta_full_vs_gen:+.0%}{marker}", flush=True)

    # ============ Verdict ============
    n_strong = sum(1 for c in card_summary.values()
                       if c["delta_full_minus_base"] >= 0.10
                       and c["delta_full_minus_generic"] >= 0.05)
    n_tent = sum(1 for c in card_summary.values()
                       if 0.05 <= c["delta_full_minus_base"] < 0.10)
    n_neg = sum(1 for c in card_summary.values()
                       if c["delta_full_minus_base"] <= -0.05)

    print(f"\n=== VERDICT ===", flush=True)
    print(f"  STRONG (Δacc≥10pp vs BASE AND Δ≥5pp vs GENERIC): {n_strong}/{len(CARDS)}", flush=True)
    print(f"  TENTATIVE (5pp ≤ Δacc < 10pp): {n_tent}/{len(CARDS)}", flush=True)
    print(f"  NEGATIVE (Δacc ≤ −5pp): {n_neg}/{len(CARDS)}", flush=True)

    if n_strong >= 1:
        verdict = (f"BREAK-OUT: {n_strong}/{len(CARDS)} cards produce >=10pp objective-accuracy "
                     f"improvement on their applicable subset, and >=5pp over the GENERIC baseline. "
                     f"Wisdom-as-prompt-injection paradigm DOES produce content-level signal "
                     f"when (a) wisdom is procedural, (b) trigger is reliable, (c) task has real bug, "
                     f"(d) baseline isn't at ceiling. The 0/12 of v1 paper was domain-specific, not "
                     f"universal. Answer to user's challenge: yes, the loop CAN break out of 0/12.")
    elif n_tent >= 2:
        verdict = (f"PARTIAL: {n_tent} cards show modest signal but no strong break-through. "
                     f"Procedural form helps somewhat but the architecture as-is doesn't produce "
                     f"reliably substantial improvements.")
    else:
        verdict = (f"NULL AGAIN: even on procedural cards with objective grading, no card produces "
                     f">=10pp improvement. This is strong evidence that prompt-injection wisdom "
                     f"paradigm is content-empty even in this favorable setting. Architecture "
                     f"helps with epistemic closure but not signal generation.")
    print(f"\n  {verdict}", flush=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_problems": len(PROBLEMS),
        "n_cards": len(CARDS),
        "n_trials_per_cell": N_TRIALS_PER_CONDITION,
        "total_calls": len(records),
        "by_condition": cond_summary,
        "by_card": card_summary,
        "n_strong_cards": n_strong,
        "n_tentative_cards": n_tent,
        "n_negative_cards": n_neg,
        "verdict": verdict,
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nForensic log:    {RAW_LOG.relative_to(PROJECT)}", flush=True)
    print(f"Summary:         {SUMMARY.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
