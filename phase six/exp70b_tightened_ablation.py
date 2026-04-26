"""Exp 70b — diagnose Sub-MVP ablation leakage.

Sub-MVP found that WITH-vs-BASE win rates were promising on quantifier and
multistep, but on the constraint slice the ABLATED condition outperformed
WITH by a huge margin. The hypothesis: the original ablation kept the
verification line (e.g. 'Re-read each numbered constraint and point to
the part of your solution that satisfies it'), which is itself a procedure
('answer first, then verify'). For hard slices like constraint puzzles, that
backend procedure was actually doing more work than the upfront procedure
in the full card.

To isolate the procedural-core effect cleanly, we re-run the ABLATED
condition with a TIGHTENED ablation that includes ONLY the trigger and
failure-mode label (no verification line, no procedure). This is the
minimal context-only baseline.

We reuse cached BASE and WITH answers from Exp 70's log (no need to
re-solve those). Only the ABLATED condition is re-generated, plus the
WITH-vs-ABLATED-tight and ABLATED-tight-vs-BASE judgments.

Cost: 5 slices x 50 problems = 250 solves + 5 x 2 pair-types x 3 judges
x 50 = 1500 judgments. ~$5, ~15 min.

INTERPRETATION
==============
If WITH still cleanly beats tightened-ABLATED on quantifier/multistep
(wr >= 0.60 with ablated below 0.55), the form's procedural core is a
real causal driver. If the gap collapses, the original Sub-MVP signal
was largely from the verification rhetoric, not the procedure.
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
from card_schema import load_cards
from slices import ALL_SLICES

PARALLEL = 8
AUTO = PROJECT / "phase six" / "autonomous"
OUT_LOG = AUTO / "exp70b_tightened_ablation_log.json"
PRIOR_LOG = AUTO / "exp70_sub_mvp_log.json"
CARDS_PATH = PROJECT / "phase six" / "cards" / "starter_cards.json"

CHAMPION_CARD = {
    "bayesian":      "denominator_lock",
    "quantifier":    "quantifier_swap_check",
    "multistep":     "step_by_step_lock",
    "constraint":    "constraint_enumeration",
    "counterfactual":"hidden_assumption_surface",
}


def render_tightened_ablation(card):
    """ABLATED-TIGHT: trigger + failure label only. No procedure, no
    verification. The minimal context-only baseline."""
    return (
        f"## METHODOLOGICAL HINT: {card.name}\n"
        f"Trigger: {card.trigger}\n"
        f"Failure to avoid: {card.failure_prevented}"
    )


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
            seed = hash(pid + label_a + label_b + jname + "tight") % (2**32)
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
    print(f"=== Exp 70b: tightened-ablation diagnostic ===", flush=True)

    # Reload cached BASE and WITH answers from Exp 70
    print(f"  Loading cached BASE+WITH answers from Exp 70...", flush=True)
    e70 = json.loads(PRIOR_LOG.read_text(encoding="utf-8"))
    cached_answers = e70["answers"]
    print(f"  Got {sum(len(cached_answers[s]['BASE']) for s in CHAMPION_CARD)} BASE answers, "
          f"{sum(len(cached_answers[s]['WITH']) for s in CHAMPION_CARD)} WITH answers", flush=True)

    cards_all = {c.name: c for c in load_cards(CARDS_PATH)}
    solver = cheap("gemini")
    judges = [
        ("gemini", cheap("gemini")),
        ("claude_haiku", cheap("claude_haiku")),
        ("gpt_mini", cheap("gpt_mini")),
    ]

    # ---- Stage 1: re-solve only ABLATED-TIGHT condition --------------
    print(f"\n[1/3] Solving ABLATED-TIGHT condition (5 slices x 50 = 250 solves)...", flush=True)
    t0 = time.time()
    answers = {}
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        card = cards_all[CHAMPION_CARD[slice_name]]
        rendered = render_tightened_ablation(card)
        answers[slice_name] = {"ABLATED_TIGHT": {}}

        # cache pull
        answers[slice_name]["BASE"] = cached_answers[slice_name]["BASE"]
        answers[slice_name]["WITH"] = cached_answers[slice_name]["WITH"]

        def solve_task(p):
            return p["pid"], solve(solver, p["prompt"], rendered)

        done = 0
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = [ex.submit(solve_task, p) for p in problems]
            for f in as_completed(futs):
                pid, ans = f.result()
                answers[slice_name]["ABLATED_TIGHT"][pid] = ans
                done += 1
                if done % 25 == 0:
                    print(f"  [{slice_name}] ABLATED-TIGHT {done}/{len(problems)} "
                          f"({time.time()-t0:.0f}s)", flush=True)
    print(f"  All ABLATED-TIGHT solves done in {time.time()-t0:.0f}s", flush=True)

    # ---- Stage 2: objective accuracy on ABLATED-TIGHT ----------------
    print(f"\n[2/3] Objective grading vs gold...", flush=True)
    obj = {}
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        b_correct = sum(score(cached_answers[slice_name]["BASE"].get(p["pid"],""),
                                p["gold"], p.get("tol", 1.05)) for p in problems)
        w_correct = sum(score(cached_answers[slice_name]["WITH"].get(p["pid"],""),
                                p["gold"], p.get("tol", 1.05)) for p in problems)
        a_correct = sum(score(answers[slice_name]["ABLATED_TIGHT"].get(p["pid"],""),
                                p["gold"], p.get("tol", 1.05)) for p in problems)
        obj[slice_name] = {
            "BASE_acc": b_correct/len(problems),
            "WITH_acc": w_correct/len(problems),
            "ABLATED_TIGHT_acc": a_correct/len(problems),
        }
        print(f"  {slice_name:14s}: BASE={obj[slice_name]['BASE_acc']:.1%} "
              f"WITH={obj[slice_name]['WITH_acc']:.1%} "
              f"ABL_TIGHT={obj[slice_name]['ABLATED_TIGHT_acc']:.1%} | "
              f"Δ(W-B)={obj[slice_name]['WITH_acc']-obj[slice_name]['BASE_acc']:+.1%} "
              f"Δ(AT-B)={obj[slice_name]['ABLATED_TIGHT_acc']-obj[slice_name]['BASE_acc']:+.1%}",
              flush=True)

    # ---- Stage 3: pairwise judging on the new ablation ---------------
    print(f"\n[3/3] Cross-family pairwise (2 pairs x 3 judges x 5 slices x 50 = "
          f"1500 judgments)...", flush=True)
    t1 = time.time()
    pairwise = {}
    for slice_name in CHAMPION_CARD:
        problems = ALL_SLICES[slice_name]
        pairwise[slice_name] = {}
        for label_a, label_b in [("ABLATED_TIGHT", "BASE"),
                                    ("WITH", "ABLATED_TIGHT")]:
            pname = f"{label_a}_vs_{label_b}"
            r = run_pairwise_panel(judges,
                                    answers[slice_name][label_a],
                                    answers[slice_name][label_b],
                                    label_a, label_b, problems)
            pairwise[slice_name][pname] = r
            print(f"  [{slice_name}] {pname:28s}: "
                  f"mean wr={r['mean_wr']:.3f} (min={r['min_wr']:.3f}) "
                  f"per-judge: gem={r['gemini']['wr_a']:.3f} "
                  f"hai={r['claude_haiku']['wr_a']:.3f} "
                  f"gpt={r['gpt_mini']['wr_a']:.3f}", flush=True)
    print(f"  Judging done in {time.time()-t1:.0f}s", flush=True)

    # ---- Re-verdict with tightened ablation --------------------------
    print(f"\n=== ACCEPTANCE VERDICT (TIGHTENED ABLATION) ===", flush=True)

    # Pull WITH-vs-BASE from prior log
    e70_pairwise = e70["pairwise_panel"]
    verdicts = {}
    for slice_name in CHAMPION_CARD:
        wr_panel_old = e70_pairwise[slice_name]["WITH_vs_BASE"]
        wr_mean_wb = wr_panel_old["mean_wr"]; wr_min_wb = wr_panel_old["min_wr"]
        # tightened ablation
        abl_tight_panel = pairwise[slice_name]["ABLATED_TIGHT_vs_BASE"]
        abl_tight_mean = abl_tight_panel["mean_wr"]
        # WITH vs ABLATED-tight (new)
        wva_tight = pairwise[slice_name]["WITH_vs_ABLATED_TIGHT"]
        wva_mean = wva_tight["mean_wr"]; wva_min = wva_tight["min_wr"]

        delta_obj = obj[slice_name]["WITH_acc"] - obj[slice_name]["BASE_acc"]

        # STRONG: WITH cleanly beats BASE *and* ablation. Both gaps must hold.
        strong = (wr_mean_wb >= 0.60 and wr_min_wb >= 0.56
                    and wva_mean >= 0.60 and wva_min >= 0.56
                    and abl_tight_mean < 0.55 and delta_obj >= 0)
        tentative = (wr_mean_wb >= 0.55 and wva_mean >= 0.55 and delta_obj >= 0)

        if strong:
            verdict = "STRONG SURVIVOR (procedural core is causal)"
        elif tentative:
            verdict = "TENTATIVE SURVIVOR"
        else:
            reasons = []
            if wr_mean_wb < 0.55: reasons.append(f"WITH-vs-BASE wr {wr_mean_wb:.3f} < 0.55")
            if wva_mean < 0.55: reasons.append(f"WITH-vs-ABL-TIGHT wr {wva_mean:.3f} < 0.55")
            if delta_obj < 0: reasons.append(f"obj Δacc {delta_obj:+.1%} negative")
            verdict = "REJECT (" + "; ".join(reasons) + ")"

        verdicts[slice_name] = {
            "card": CHAMPION_CARD[slice_name],
            "verdict": verdict,
            "wr_with_vs_base_mean": wr_mean_wb,
            "wr_ablated_tight_vs_base_mean": abl_tight_mean,
            "wr_with_vs_ablated_tight_mean": wva_mean,
            "wr_with_vs_ablated_tight_min": wva_min,
            "delta_obj_acc": delta_obj,
        }
        print(f"  {slice_name:14s}: {verdict}", flush=True)
        print(f"    WITH-vs-BASE       mean={wr_mean_wb:.3f}", flush=True)
        print(f"    ABL_TIGHT-vs-BASE  mean={abl_tight_mean:.3f}", flush=True)
        print(f"    WITH-vs-ABL_TIGHT  mean={wva_mean:.3f} (min={wva_min:.3f})", flush=True)

    n_strong = sum(1 for v in verdicts.values() if "STRONG" in v["verdict"])
    n_tent = sum(1 for v in verdicts.values() if "TENTATIVE" in v["verdict"])
    print(f"\n=== TIGHTENED-ABLATION RESULT: {n_strong}/5 STRONG, {n_tent}/5 TENTATIVE ===", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "champion_card_per_slice": CHAMPION_CARD,
        "objective_accuracy": obj,
        "pairwise": pairwise,
        "verdicts": verdicts,
        "n_strong": n_strong, "n_tentative": n_tent,
        "answers_ablated_tight": {s: answers[s]["ABLATED_TIGHT"] for s in CHAMPION_CARD},
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}", flush=True)


if __name__ == "__main__":
    main()
