"""Exp 70d — Final specificity check: does worked-example presence
make trigger+failure-label specifically helpful?

Exp 70c found that ABL_TIGHT (trigger+failure-label only) is matched
or beaten by GENERIC_WARNING ("be careful, watch for hasty
conclusions"). The specificity of the trigger and failure-label
contributed nothing beyond a generic 'slow-down' / 'think harder'
prefix.

But Exp 68 had wr=0.846 from a Bayesian template that included
TWO WORKED EXAMPLES. Exp 70b/c stripped examples out. Maybe
the worked example is the missing factor: the trigger+failure-label
becomes specifically useful only when a class-matched worked example
shows the procedure.

To test, we add two new conditions on the same 5 slices, 50 problems
each:

  TIGHT_WITH_EX      : original trigger + failure-label + the card's
                        specific worked example (already in starter_cards.json)
  GENERIC_WITH_EX    : generic warning + a generic worked example
                        (compound-interest arithmetic, length-matched)

Comparisons (3 judges x 50 problems each):
  TIGHT_WITH_EX vs GENERIC_WITH_EX  — does specificity matter when both have an example?
  TIGHT_WITH_EX vs GENERIC          — does adding a specific example over plain generic help?

If TIGHT_WITH_EX cleanly beats GENERIC_WITH_EX (mean wr >= 0.60), the
worked-example was the missing factor and v2 still has a path.
If they tie, specificity doesn't matter even with worked examples;
the wisdom-content framework is broadly invalidated.

Cost: 2 conditions x 250 problems = 500 solves + 2 pair-types x 3
judges x 50 x 5 = 1500 judgments. ~$6, ~20 min.
"""
import json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase six" / "cards"))
sys.path.insert(0, str(PROJECT / "phase six" / "slices"))

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
        if name == "RUOLI_GPT_KEY": os.environ.setdefault("GPT5_API_KEY", val)
        if name == "RUOLI_CLAUDE_KEY": os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)
_load_api_keys()

from model_router import cheap
from cached_framework import judge_pair, _save_content_cache
from card_schema import load_cards
from slices import ALL_SLICES

PARALLEL = 8
AUTO = PROJECT / "phase six" / "autonomous"
OUT_LOG = AUTO / "exp70d_worked_example_check_log.json"
PRIOR70C = AUTO / "exp70c_specificity_check_log.json"
CARDS_PATH = PROJECT / "phase six" / "cards" / "starter_cards.json"

CHAMPION_CARD = {
    "bayesian":      "denominator_lock",
    "quantifier":    "quantifier_swap_check",
    "multistep":     "step_by_step_lock",
    "constraint":    "constraint_enumeration",
    "counterfactual":"hidden_assumption_surface",
}

# Generic worked example: a compound-interest computation.
# Class-matched only on multistep slice; unmatched on others (acts as
# 'show me you can think step by step' without giving slice-specific
# scaffolding).
GENERIC_WARNING_WITH_EX = (
    "## METHODOLOGICAL HINT: careful_reasoning\n"
    "Trigger: This problem may be tricky.\n"
    "Failure to avoid: Hasty conclusions; missing important details.\n"
    "Example of careful reasoning: To compute $1000 at 5% compounded "
    "over 2 years: Step 1: After year 1, balance = 1000 * 1.05 = 1050. "
    "Step 2: After year 2, balance = 1050 * 1.05 = 1102.50. "
    "Final answer: $1102.50."
)


def render_tight_with_example(card):
    """Trigger + failure-label + the card's specific worked example.

    No procedural steps, no verification. Cleanly compares against
    GENERIC_WARNING_WITH_EX which has a generic example."""
    parts = [
        f"## METHODOLOGICAL HINT: {card.name}",
        f"Trigger: {card.trigger}",
        f"Failure to avoid: {card.failure_prevented}",
    ]
    if card.example:
        parts.append(f"Example: {card.example}")
    return "\n".join(parts)


SOLVE_PROMPT = """{card_section}## Problem
{problem}

## Output
Reason step by step concisely (3-8 sentences), then on the LAST LINE write exactly:
ANSWER: <answer>
"""


def solve(client, problem_text, card_render):
    card_section = (card_render + "\n\n") if card_render else ""
    try:
        r = client.generate(
            SOLVE_PROMPT.format(card_section=card_section, problem=problem_text),
            max_tokens=900, temperature=0.0,
        )
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"


def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()


def score(answer_text, gold, tol):
    raw = extract_answer(answer_text)
    if isinstance(gold, str):
        return 1 if gold.lower().strip() in raw.lower() else 0
    s = raw.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    nm = re.search(r'-?\d+(?:\.\d+)?(?:[eE]-?\d+)?', s)
    if not nm: return 0
    try: val = float(nm.group())
    except Exception: return 0
    if not isinstance(tol, (int, float)) or tol >= 1.001:
        if gold == 0: return 1 if abs(val) < 0.001 else 0
        rel = abs(val - gold) / abs(gold)
        return 1 if rel <= ((tol or 1.05) - 1.0) else 0
    return 1 if abs(val - gold) <= tol else 0


def judge_with_side_randomize(client, problem, ans_a, ans_b, label_a, label_b, seed):
    rng = random.Random(seed)
    if rng.random() < 0.5:
        left, right, a_was = ans_a, ans_b, "A"
    else:
        left, right, a_was = ans_b, ans_a, "B"
    try:
        v = judge_pair(client, problem, left, right)
        w = v.get("winner", "tie")
        if w == "tie": return "tie"
        return label_a if w == a_was else label_b
    except Exception:
        return "tie"


def run_pairwise_panel(judges, ans_dict_a, ans_dict_b, label_a, label_b, problems):
    results = {}
    for jname, jclient in judges:
        wins_a = wins_b = ties = 0
        per_pid = {}
        def one(p):
            pid = p["pid"]
            a = ans_dict_a.get(pid, ""); b = ans_dict_b.get(pid, "")
            if not a or not b: return pid, "tie"
            seed = hash(pid + label_a + label_b + jname + "wex") % (2**32)
            return pid, judge_with_side_randomize(jclient, p["prompt"], a, b, label_a, label_b, seed)
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(one, p) for p in problems]
            for f in as_completed(futs):
                pid, w = f.result()
                per_pid[pid] = w
                if w == label_a: wins_a += 1
                elif w == label_b: wins_b += 1
                else: ties += 1
        n_eff = wins_a + wins_b
        wr = wins_a / n_eff if n_eff else 0.5
        results[jname] = {"wins_a": wins_a, "wins_b": wins_b, "ties": ties,
                              "n_eff": n_eff, "wr_a": wr, "per_pid": per_pid}
    _save_content_cache()
    wrs = [results[j]["wr_a"] for j in [j for j, _ in judges]]
    results["mean_wr"] = sum(wrs) / len(wrs)
    results["min_wr"] = min(wrs)
    results["max_wr"] = max(wrs)
    return results


def main():
    print(f"=== Exp 70d: worked-example specificity check ===", flush=True)

    e70c = json.loads(PRIOR70C.read_text(encoding="utf-8"))
    generic_answers = e70c["answers_generic"]
    print(f"  Loaded GENERIC (no-example) cached answers from Exp 70c", flush=True)

    cards_all = {c.name: c for c in load_cards(CARDS_PATH)}
    solver = cheap("gemini")
    judges = [
        ("gemini", cheap("gemini")),
        ("claude_haiku", cheap("claude_haiku")),
        ("gpt_mini", cheap("gpt_mini")),
    ]

    # ---- Stage 1: solve TIGHT_WITH_EX and GENERIC_WITH_EX ------------
    print(f"\n[1/3] Solving TIGHT_WITH_EX + GENERIC_WITH_EX (2 x 5 x 50 = 500 solves)...", flush=True)
    t0 = time.time()
    tight_ex_answers = {}
    gen_ex_answers = {}
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        card = cards_all[CHAMPION_CARD[slice_name]]
        tight_ex_render = render_tight_with_example(card)
        tight_ex_answers[slice_name] = {}
        gen_ex_answers[slice_name] = {}

        def solve_task(cond, p):
            render = tight_ex_render if cond == "TIGHT_EX" else GENERIC_WARNING_WITH_EX
            return cond, p["pid"], solve(solver, p["prompt"], render)

        tasks = [(c, p) for c in ["TIGHT_EX", "GENERIC_EX"] for p in problems]
        done = 0
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(solve_task, c, p) for c, p in tasks]
            for f in as_completed(futs):
                c, pid, ans = f.result()
                if c == "TIGHT_EX": tight_ex_answers[slice_name][pid] = ans
                else: gen_ex_answers[slice_name][pid] = ans
                done += 1
                if done % 25 == 0:
                    print(f"  [{slice_name}] {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  All solves done in {time.time()-t0:.0f}s", flush=True)

    # ---- Stage 2: objective accuracy ---------------------------------
    print(f"\n[2/3] Objective grading vs gold...", flush=True)
    obj = {}
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        g = sum(score(generic_answers[slice_name].get(p["pid"],""),
                       p["gold"], p.get("tol", 1.05)) for p in problems)
        te = sum(score(tight_ex_answers[slice_name].get(p["pid"],""),
                       p["gold"], p.get("tol", 1.05)) for p in problems)
        ge = sum(score(gen_ex_answers[slice_name].get(p["pid"],""),
                       p["gold"], p.get("tol", 1.05)) for p in problems)
        obj[slice_name] = {"GENERIC_acc": g/len(problems),
                              "TIGHT_EX_acc": te/len(problems),
                              "GENERIC_EX_acc": ge/len(problems)}
        print(f"  {slice_name:14s}: GENERIC={obj[slice_name]['GENERIC_acc']:.1%} "
              f"TIGHT_EX={obj[slice_name]['TIGHT_EX_acc']:.1%} "
              f"GENERIC_EX={obj[slice_name]['GENERIC_EX_acc']:.1%}", flush=True)

    # ---- Stage 3: pairwise --------------------------------------------
    print(f"\n[3/3] Pairwise judging (2 pairs x 3 judges x 5 slices x 50 = 1500 judgments)...", flush=True)
    t1 = time.time()
    pairwise = {}
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        pairwise[slice_name] = {}
        for label_a, label_b, ans_a, ans_b in [
            ("TIGHT_EX", "GENERIC_EX", tight_ex_answers[slice_name], gen_ex_answers[slice_name]),
            ("TIGHT_EX", "GENERIC", tight_ex_answers[slice_name], generic_answers[slice_name]),
        ]:
            pname = f"{label_a}_vs_{label_b}"
            r = run_pairwise_panel(judges, ans_a, ans_b, label_a, label_b, problems)
            pairwise[slice_name][pname] = r
            print(f"  [{slice_name:14s}] {pname:25s}: mean wr={r['mean_wr']:.3f} "
                  f"(min={r['min_wr']:.3f}) gem={r['gemini']['wr_a']:.3f} "
                  f"hai={r['claude_haiku']['wr_a']:.3f} gpt={r['gpt_mini']['wr_a']:.3f}", flush=True)
    print(f"  Judging done in {time.time()-t1:.0f}s", flush=True)

    # ---- Verdict per slice ------------------------------------------
    print(f"\n=== WORKED-EXAMPLE SPECIFICITY VERDICT PER SLICE ===", flush=True)
    verdicts = {}
    for slice_name in CHAMPION_CARD:
        delta_obj_te_ge = obj[slice_name]["TIGHT_EX_acc"] - obj[slice_name]["GENERIC_EX_acc"]
        wr_te_vs_ge = pairwise[slice_name]["TIGHT_EX_vs_GENERIC_EX"]["mean_wr"]
        wr_te_vs_g = pairwise[slice_name]["TIGHT_EX_vs_GENERIC"]["mean_wr"]

        if delta_obj_te_ge >= 0.10 and wr_te_vs_ge >= 0.60:
            v = "SPECIFIC: trigger+failure-label+example is content-causal"
        elif wr_te_vs_ge >= 0.55 and delta_obj_te_ge >= 0.05:
            v = "TENTATIVE SPECIFIC: signal exists but small"
        elif abs(delta_obj_te_ge) < 0.05 and 0.45 <= wr_te_vs_ge <= 0.55:
            v = "NOT SPECIFIC: examples don't make trigger+failure causal"
        else:
            v = f"MIXED (Δacc={delta_obj_te_ge:+.1%}, wr_TE_vs_GE={wr_te_vs_ge:.3f})"
        verdicts[slice_name] = {
            "verdict": v,
            "delta_acc_TE_minus_GE": delta_obj_te_ge,
            "wr_TE_vs_GE": wr_te_vs_ge,
            "wr_TE_vs_G": wr_te_vs_g,
        }
        print(f"  {slice_name:14s}: {v}", flush=True)
        print(f"    Δacc(TIGHT_EX - GENERIC_EX) = {delta_obj_te_ge:+.1%}", flush=True)
        print(f"    wr(TIGHT_EX vs GENERIC_EX)  = {wr_te_vs_ge:.3f}", flush=True)
        print(f"    wr(TIGHT_EX vs GENERIC)     = {wr_te_vs_g:.3f}  (any gain over no-example?)", flush=True)

    n_specific = sum(1 for v in verdicts.values() if "SPECIFIC" in v["verdict"] and "TENTATIVE" not in v["verdict"] and "NOT" not in v["verdict"])
    n_tent = sum(1 for v in verdicts.values() if "TENTATIVE" in v["verdict"])
    n_not = sum(1 for v in verdicts.values() if "NOT SPECIFIC" in v["verdict"])
    print(f"\n=== RESULT: {n_specific}/5 SPECIFIC (with examples), "
          f"{n_tent}/5 TENTATIVE, {n_not}/5 NOT SPECIFIC ===", flush=True)
    if n_specific >= 1 or n_tent >= 2:
        print("  → Worked example is the missing factor; v2 has a path "
              "with format = trigger + failure-label + worked example.", flush=True)
    else:
        print("  → Worked examples DO NOT recover specificity. The wisdom-as-prompt-injection "
              "framework is broadly invalidated. Path A: write v2 paper as a deeper negative result.", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "objective_accuracy": obj,
        "pairwise": pairwise,
        "verdicts": verdicts,
        "answers_tight_ex": tight_ex_answers,
        "answers_generic_ex": gen_ex_answers,
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}", flush=True)


if __name__ == "__main__":
    main()
