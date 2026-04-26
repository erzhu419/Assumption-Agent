"""Exp 70 — Loop v2 Sub-MVP: do hand-written micro-procedure cards survive
audit on slice-targeted problems?

This is the absolute first test of the redesigned loop. We are not yet
generating cards; we just check whether the FORM of the wisdom (a
triggered cognitive patch with worked example) is, by itself, capable
of producing audit-surviving improvements.

DESIGN
======
Per slice, each problem is solved THREE ways:
  - BASE        : no card injected, free-form solve
  - WITH-CARD   : the slice-specific best card injected before the problem
  - ABLATED     : the same card with its procedural core removed
                  (keeps trigger + verification only)

For each card x slice:
  1. Objective accuracy on the slice's problems (50): BASE / WITH / ABLATED
  2. Cross-family pairwise A/B: WITH-vs-BASE, ABLATED-vs-BASE, WITH-vs-ABLATED
     using gemini-3-flash + claude-haiku-4.5 + gpt-5.4-mini as a 3-judge panel
  3. Acceptance:
     STRONG     : mean wr_with_vs_base >= 0.60 AND no judge below 0.56
                  AND ablated wr < 0.55 AND obj_acc(WITH) - obj_acc(BASE) >= 0
     TENTATIVE  : mean wr_with_vs_base >= 0.55 AND no judge below 0.51
                  AND ablated wr < mean wr_with - 0.05
                  AND obj_acc(WITH) - obj_acc(BASE) >= 0

If any starter card meets STRONG criteria on its targeted slice, the
form works. If only TENTATIVE, the form has signal but is fragile.
If neither, the form is wrong even before we add generation.

For Sub-MVP we restrict to ONE card per slice (the most likely to work)
to keep cost under ~$8: 5 cards x 3 conditions x 50 problems = 750 solves
+ 5 cards x 3 pair-types x 3 judges x 50 = 2250 judgments.

ESTIMATED COST: ~$10-15, ~30-60 min wall.
"""
import json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

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
from card_schema import load_cards, Card
from slices import ALL_SLICES, all_problems

PARALLEL = 8
AUTO = PROJECT / "phase six" / "autonomous"
OUT_LOG = AUTO / "exp70_sub_mvp_log.json"
CARDS_PATH = PROJECT / "phase six" / "cards" / "starter_cards.json"

# ============================================================
# Sub-MVP scope: pick ONE strongest-bet card per slice
# ============================================================
CHAMPION_CARD = {
    "bayesian":      "denominator_lock",
    "quantifier":    "quantifier_swap_check",
    "multistep":     "step_by_step_lock",
    "constraint":    "constraint_enumeration",
    "counterfactual":"hidden_assumption_surface",
}

SOLVE_PROMPT = """{card_section}## Problem
{problem}

## Output
Reason step by step concisely (3-8 sentences), then on the LAST LINE write exactly:
ANSWER: <answer>
"""


def solve(client, problem_text, card_render):
    """card_render is empty string for BASE, or full/ablated card for the other conditions."""
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
    """Score against gold. tol >= 1.001 means multiplicative; else absolute."""
    raw = extract_answer(answer_text)
    if isinstance(gold, str):
        return 1 if gold.lower().strip() in raw.lower() else 0
    s = raw.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    nm = re.search(r'-?\d+(?:\.\d+)?(?:[eE]-?\d+)?', s)
    if not nm: return 0
    try: val = float(nm.group())
    except Exception: return 0
    if not isinstance(tol, (int, float)) or tol >= 1.001:
        # multiplicative: tol=1.05 means within 5%
        if gold == 0:
            return 1 if abs(val) < 0.001 else 0
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
    """Run 3-judge panel on a set of pairs. Returns per-judge wr + mean."""
    results = {}
    for jname, jclient in judges:
        wins_a = wins_b = ties = 0
        per_pid = {}
        def one(p):
            pid = p["pid"]
            a = ans_dict_a.get(pid, ""); b = ans_dict_b.get(pid, "")
            if not a or not b: return pid, "tie"
            seed = hash(pid + label_a + label_b + jname) % (2**32)
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
    print(f"=== Exp 70: Loop v2 Sub-MVP — does the card form work? ===", flush=True)

    cards_all = {c.name: c for c in load_cards(CARDS_PATH)}
    print(f"  Loaded {len(cards_all)} cards", flush=True)
    print(f"  Champion card per slice:", flush=True)
    for slice_name, card_name in CHAMPION_CARD.items():
        print(f"    {slice_name:14s} → {card_name}", flush=True)
    print()

    solver = cheap("gemini")
    judges = [
        ("gemini", cheap("gemini")),
        ("claude_haiku", cheap("claude_haiku")),
        ("gpt_mini", cheap("gpt_mini")),
    ]

    # ---- Stage 1: solve all 5 slices x 3 conditions ------------------
    print(f"[1/3] Solving 5 slices x 3 conditions x 50 problems = "
          f"{5*3*50} solves...", flush=True)
    t0 = time.time()
    answers = {}  # answers[slice][cond][pid] -> string
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        card = cards_all[CHAMPION_CARD[slice_name]]
        rendered_full = card.render_full()
        rendered_abl = card.render_ablated()
        answers[slice_name] = {"BASE": {}, "WITH": {}, "ABLATED": {}}

        def solve_task(cond, p):
            if cond == "BASE":
                return cond, p["pid"], solve(solver, p["prompt"], "")
            elif cond == "WITH":
                return cond, p["pid"], solve(solver, p["prompt"], rendered_full)
            else:
                return cond, p["pid"], solve(solver, p["prompt"], rendered_abl)

        tasks = [(c, p) for c in ["BASE", "WITH", "ABLATED"] for p in problems]
        done = 0
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(solve_task, c, p) for c, p in tasks]
            for f in as_completed(futs):
                c, pid, ans = f.result()
                answers[slice_name][c][pid] = ans
                done += 1
                if done % 50 == 0:
                    print(f"  [{slice_name}] {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  All solves done in {time.time()-t0:.0f}s", flush=True)

    # ---- Stage 2: objective grading ----------------------------------
    print(f"\n[2/3] Objective grading vs gold...", flush=True)
    obj = {}
    per_pid_correct = {}  # per_pid_correct[slice][cond][pid] -> 0/1
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        per_pid_correct[slice_name] = {}
        obj[slice_name] = {}
        for cond in ["BASE", "WITH", "ABLATED"]:
            per_pid_correct[slice_name][cond] = {}
            n_correct = 0
            for p in problems:
                ans = answers[slice_name][cond].get(p["pid"], "")
                sc = score(ans, p["gold"], p.get("tol", 1.05))
                per_pid_correct[slice_name][cond][p["pid"]] = sc
                n_correct += sc
            obj[slice_name][cond] = {"correct": n_correct, "total": len(problems),
                                       "acc": n_correct / len(problems)}
        b = obj[slice_name]["BASE"]["acc"]
        w = obj[slice_name]["WITH"]["acc"]
        a = obj[slice_name]["ABLATED"]["acc"]
        print(f"  {slice_name:14s}: BASE={b:.1%} WITH={w:.1%} ABLATED={a:.1%} | "
                f"Δ(W-B)={w-b:+.1%} Δ(A-B)={a-b:+.1%}", flush=True)

    # ---- Stage 3: cross-family pairwise judging ----------------------
    print(f"\n[3/3] Cross-family pairwise judging (3 pair-types x 3 judges x 5 slices x 50 = "
          f"{3*3*5*50} judgments)...", flush=True)
    t1 = time.time()
    pairwise = {}
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        pairwise[slice_name] = {}
        for label_a, label_b in [("WITH","BASE"), ("ABLATED","BASE"), ("WITH","ABLATED")]:
            pname = f"{label_a}_vs_{label_b}"
            r = run_pairwise_panel(judges,
                                    answers[slice_name][label_a],
                                    answers[slice_name][label_b],
                                    label_a, label_b, problems)
            pairwise[slice_name][pname] = r
            print(f"  [{slice_name}] {pname:18s}: "
                    f"mean wr={r['mean_wr']:.3f} (min={r['min_wr']:.3f} max={r['max_wr']:.3f}) "
                    f"per-judge: gemini={r['gemini']['wr_a']:.3f} "
                    f"haiku={r['claude_haiku']['wr_a']:.3f} "
                    f"gpt={r['gpt_mini']['wr_a']:.3f}", flush=True)
    print(f"  Judging done in {time.time()-t1:.0f}s", flush=True)

    # ---- Acceptance verdict per card --------------------------------
    print(f"\n=== ACCEPTANCE VERDICT PER CARD ===", flush=True)
    verdicts = {}
    for slice_name in CHAMPION_CARD:
        b = obj[slice_name]["BASE"]["acc"]
        w = obj[slice_name]["WITH"]["acc"]
        delta_obj = w - b
        wr_panel = pairwise[slice_name]["WITH_vs_BASE"]
        wr_mean = wr_panel["mean_wr"]; wr_min = wr_panel["min_wr"]
        abl_panel = pairwise[slice_name]["ABLATED_vs_BASE"]
        abl_mean = abl_panel["mean_wr"]

        strong = (wr_mean >= 0.60 and wr_min >= 0.56
                    and abl_mean < 0.55 and delta_obj >= 0)
        tentative = (wr_mean >= 0.55 and wr_min >= 0.51
                      and abl_mean < (wr_mean - 0.05)
                      and delta_obj >= 0)

        if strong:
            verdict = "STRONG SURVIVOR"
        elif tentative:
            verdict = "TENTATIVE SURVIVOR"
        else:
            reasons = []
            if wr_mean < 0.55: reasons.append(f"mean wr {wr_mean:.3f} < 0.55")
            if wr_min < 0.51: reasons.append(f"min wr {wr_min:.3f} < 0.51")
            if abl_mean >= (wr_mean - 0.05): reasons.append(
                f"ablation wr {abl_mean:.3f} not sufficiently below ({wr_mean - 0.05:.3f})")
            if delta_obj < 0: reasons.append(f"obj Δacc {delta_obj:+.1%} negative")
            verdict = "REJECT (" + "; ".join(reasons) + ")"

        verdicts[slice_name] = {
            "card": CHAMPION_CARD[slice_name],
            "verdict": verdict,
            "wr_mean": wr_mean, "wr_min": wr_min,
            "ablated_wr_mean": abl_mean,
            "delta_obj_acc": delta_obj,
        }
        print(f"  {slice_name:14s} ({CHAMPION_CARD[slice_name]:30s}): {verdict}", flush=True)

    # ---- Final summary -----------------------------------------------
    n_strong = sum(1 for v in verdicts.values() if "STRONG" in v["verdict"])
    n_tent = sum(1 for v in verdicts.values() if "TENTATIVE" in v["verdict"])
    n_total = len(verdicts)
    print(f"\n=== SUB-MVP RESULT: {n_strong}/{n_total} STRONG, "
            f"{n_tent}/{n_total} TENTATIVE survivors ===", flush=True)
    if n_strong >= 1:
        print("  → CARD FORM WORKS. Loop v2 has a path. Proceed to MVP "
                "(failure-mining + candidate generation).", flush=True)
    elif n_tent >= 1:
        print("  → CARD FORM HAS SIGNAL but fragile. Investigate why ablation "
                "isn't sharply distinguished; tighten cards before generation.", flush=True)
    else:
        print("  → CARD FORM DOES NOT WORK at this design. Diagnose: card "
                "specificity, slice headroom, or solver/scaffold issue.", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "champion_card_per_slice": CHAMPION_CARD,
        "objective_accuracy": obj,
        "pairwise_panel": pairwise,
        "verdicts": verdicts,
        "n_strong_survivors": n_strong,
        "n_tentative_survivors": n_tent,
        "n_total": n_total,
        "answers": answers,
        "per_pid_correct": per_pid_correct,
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}", flush=True)


if __name__ == "__main__":
    main()
