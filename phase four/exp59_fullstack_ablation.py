"""Exp 59 — End-to-end ablation: per-stage contribution of the agent stack.

To support the §Roadmap claim that ``priors + world model + scheduler
+ experience'' are the necessary architecture, this experiment compares
seven conditions on Exp 56's 60-problem 4-family dataset:

  C0. baseline           : no prior given
  C1. random-pick        : uniformly pick one of {decompose, restate, estimate, constraints}
  C2. best-single        : always use the single most-helpful prior
                            (decompose, the strongest fixed in Exp 56)
  C3. Stage-1 only       : LLM scheduler picks prior (Exp 56 scheduler)
  C4. Stage-0.5 only     : world-model argmax picks prior (Exp 57 wm)
  C5. Stage-1 + Stage-2  : experience-augmented scheduler (Exp 58 exp-sched)
  C6. FULL STACK         : experience-augmented scheduler with
                            world-model as a screen --- if WM's top pick
                            disagrees with scheduler's pick AND the WM
                            top pick has higher predicted P(success),
                            override scheduler's pick

If C6 > C5 > {C3, C4} > C2 > C1 > C0, the per-stage contribution is
empirically validated: each architectural piece contributes
incremental accuracy.

Cost: only C0, C1, C6 need fresh computation; C2-C5 are reused from
Exp 56-58. ~$2.
"""
import json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
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
            os.environ.setdefault("CLAUDE_PROXY_BASE_URL", base); os.environ.setdefault("GPT5_BASE_URL", base); os.environ.setdefault("GEMINI_PROXY_BASE_URL", base)
        if name == "RUOLI_GEMINI_KEY": os.environ.setdefault("GEMINI_PROXY_API_KEY", val)
        if name == "RUOLI_GPT_KEY": os.environ.setdefault("GPT5_API_KEY", val)
        if name == "RUOLI_CLAUDE_KEY": os.environ.setdefault("CLAUDE_PROXY_API_KEY", val)
_load_api_keys()
from model_router import cheap

PARALLEL = 6
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp59_fullstack_ablation_log.json"

PRIORS = {
    "decompose": "Before answering, decompose into atomic substeps.",
    "restate":   "Before answering, RE-READ the question carefully; check if the obvious interpretation is a trap.",
    "estimate":  "Before answering, give an order-of-magnitude estimate; sanity-check final answer.",
    "constraints": "Before answering, enumerate explicit constraints; satisfy all simultaneously.",
    "none":      "",
}

SOLVE_PROMPT = """## Problem
{problem}

## Approach hint
{approach}

## Output
Reason step by step in 1-3 sentences, then on the LAST LINE write exactly:
ANSWER: <your final answer>
"""

EXP_SCHED_PROMPT = """You are a strategy selector. Past examples of (problem, picked_prior,
was_correct) inform your choice.

## Past examples
{examples}

## New problem
{problem}

## Output (JSON)
{{"choice": "decompose"|"restate"|"estimate"|"constraints"|"none", "reason": "..."}}
"""

def solve(client, problem, prior_name):
    p = PRIORS[prior_name]
    approach = f"Use this strategy: {p}" if p else "Use any approach you think appropriate."
    try:
        r = client.generate(SOLVE_PROMPT.format(problem=problem, approach=approach),
                             max_tokens=600, temperature=0.0)
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"

def exp_schedule(client, problem, training_triples, k=15):
    rng = random.Random(hash(problem) % (2**32))
    sampled = rng.sample(training_triples, min(k, len(training_triples)))
    examples_text = "\n".join(
        f"- problem: \"{t['problem'][:150]}{'...' if len(t['problem'])>150 else ''}\"\n"
        f"  picked_prior: {t['prior']}\n"
        f"  was_correct: {bool(t['outcome'])}"
        for t in sampled)
    try:
        r = client.generate(EXP_SCHED_PROMPT.format(examples=examples_text, problem=problem),
                             max_tokens=300, temperature=0.0)
        m = re.search(r'"choice"\s*:\s*"(decompose|restate|estimate|constraints|none)"', r["text"])
        if m: return m.group(1)
    except Exception: pass
    return "none"

def extract_answer(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()

def score_numeric(ext, gold, tol=None):
    s = ext.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m: return 0
    try: ev = float(m.group())
    except: return 0
    try: gv = float(gold)
    except: return 0
    if tol is not None:
        if gv == 0: return 1 if ev == 0 else 0
        ratio = abs(ev / gv) if gv != 0 else float('inf')
        return 1 if (1.0 / tol) <= ratio <= tol else 0
    return 1 if abs(ev - gv) < 0.02 else 0

def score_text(ext, gold):
    e = ext.lower().strip().rstrip(".")
    g = str(gold).lower().strip()
    if g in e: return 1
    g_words = set(re.findall(r'\w+', g))
    meaningful = {w for w in g_words if len(w) >= 3}
    if not meaningful: return 0
    e_words = set(re.findall(r'\w+', e))
    return 1 if len(meaningful & e_words) / len(meaningful) >= 0.5 else 0

def score(ext, gold, tol=None):
    if isinstance(gold, (int, float)): return score_numeric(ext, gold, tol)
    return score_text(ext, gold)


def main():
    print(f"=== Exp 59: full-stack ablation, per-stage contribution ===")
    e56 = json.loads((AUTO / "exp56_stage1_broader_log.json").read_text())
    e57 = json.loads((AUTO / "exp57_stage05_worldmodel_log.json").read_text())
    pp56 = e56["per_problem"]
    pp57 = e57["per_problem_picks"]

    # Reconstruct problem objects
    all_problems = []
    for pid, d in pp56.items():
        all_problems.append({"pid": pid, "prompt": d["prompt"], "gold": d["gold"],
                             "family": d["family"],
                             "optimal_prior": {"A_decompose": "decompose",
                                                 "B_restate": "restate",
                                                 "C_estimate": "estimate",
                                                 "D_constraints": "constraints"}[d["family"]],
                             "tol": 10 if d["family"] == "C_estimate" else None})

    # Stratified train/test (same split as Exp 58)
    rng = random.Random(2026)
    fams = ["A_decompose", "B_restate", "C_estimate", "D_constraints"]
    train, test = [], []
    for f in fams:
        fam_p = [p for p in all_problems if p["family"] == f]
        rng.shuffle(fam_p)
        train += fam_p[:10]; test += fam_p[10:15]
    print(f"  Test set: {len(test)} (5 per family)\n")

    # Build training triples (same as Exp 58)
    training_triples = []
    for p in train:
        d = pp56[p["pid"]]
        sched_pick = e56["scheduler_picks"].get(p["pid"], "none")
        sched_score = d.get("scheduler_score", 0)
        training_triples.append({"problem": p["prompt"], "prior": sched_pick,
                                  "outcome": int(sched_score),
                                  "family": p["family"]})

    client = cheap("gemini")

    # ---- Reuse picks/scores from Exp 56-58 for C2-C5; compute fresh for C0, C1, C6 ----
    print(f"Computing condition picks for the test set...\n")

    # C0: baseline (already in Exp 56's "baseline")
    c0_score = sum(pp56[p["pid"]].get("baseline_score", 0) for p in test)

    # C1: random-pick (compute fresh — pick uniformly from 4 priors per problem)
    c1_picks = {}
    rng_c1 = random.Random(202611)
    priors_choices = ["decompose", "restate", "estimate", "constraints"]
    for p in test:
        c1_picks[p["pid"]] = rng_c1.choice(priors_choices)

    # C2: best-single-prior on test set across families. Use Exp 56's data:
    # which fixed prior has the highest TEST-set accuracy?
    fixed_priors = ["decompose", "restate", "estimate", "constraints"]
    test_acc_per_prior = {pr: sum(pp56[p["pid"]].get(f"{pr}_score", 0) for p in test) / len(test)
                            for pr in fixed_priors}
    best_single = max(test_acc_per_prior.items(), key=lambda x: x[1])[0]
    c2_score = sum(pp56[p["pid"]].get(f"{best_single}_score", 0) for p in test)

    # C3: Stage-1 only (Exp 56 scheduler) on test
    c3_score = sum(pp56[p["pid"]].get("scheduler_score", 0) for p in test)

    # C4: Stage-0.5 only (Exp 57 world-model argmax) on test
    # Note: pp57 holds {pid: {world_model_pick, ...}}. Use the actuals scores.
    c4_score = 0
    for p in test:
        wm_pick = pp57[p["pid"]]["world_model_pick"]
        c4_score += pp57[p["pid"]]["actuals"][wm_pick]

    # C5 + C6: need fresh runs
    print(f"[1/3] Random-pick solver (C1: {len(test)} calls)...")
    c1_answers = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, client, p["prompt"], c1_picks[p["pid"]]): p for p in test}
        for f in as_completed(futs):
            p = futs[f]; c1_answers[p["pid"]] = f.result()
    c1_score_fresh = sum(score(extract_answer(c1_answers[p["pid"]]), p["gold"], p.get("tol"))
                          for p in test)

    # C5: experience-augmented scheduler (Exp 58 already did this on test set)
    e58_path = AUTO / "exp58_stage2_experience_log.json"
    if e58_path.exists():
        e58 = json.loads(e58_path.read_text())
        c5_score = round(e58.get("exp_solver_accuracy", 0) * len(test))
        c5_picks = e58.get("exp_picks", {})
    else:
        # Compute fresh
        print(f"[2/3] Exp-augmented scheduler (C5: {len(test)} sched + {len(test)} solve)...")
        c5_picks = {}
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = {ex.submit(exp_schedule, client, p["prompt"], training_triples): p for p in test}
            for f in as_completed(futs):
                p = futs[f]; c5_picks[p["pid"]] = f.result()
        c5_answers = {}
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = {ex.submit(solve, client, p["prompt"], c5_picks[p["pid"]]): p for p in test}
            for f in as_completed(futs):
                p = futs[f]; c5_answers[p["pid"]] = f.result()
        c5_score = sum(score(extract_answer(c5_answers[p["pid"]]), p["gold"], p.get("tol"))
                        for p in test)

    # C6: full stack — experience-augmented scheduler picks AUGMENTED by world model
    # If WM's argmax disagrees with scheduler's pick AND WM's confidence in its pick
    # is high, override.
    print(f"\n[3/3] Full stack (C6: {len(test)} schedule + override + solve)...")
    c6_picks = {}
    OVERRIDE_THRESHOLD = 0.7
    for p in test:
        sched_pick = c5_picks.get(p["pid"], "none")
        wm_picks = pp57[p["pid"]]["preds"]  # dict {prior: predicted_p_success}
        wm_argmax_prior = max(wm_picks.items(), key=lambda x: x[1])[0]
        wm_argmax_p = wm_picks[wm_argmax_prior]
        # Override if WM is confident AND disagrees
        if (wm_argmax_prior != sched_pick) and (wm_argmax_p > OVERRIDE_THRESHOLD):
            c6_picks[p["pid"]] = wm_argmax_prior
        else:
            c6_picks[p["pid"]] = sched_pick

    c6_answers = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, client, p["prompt"], c6_picks[p["pid"]]): p for p in test}
        for f in as_completed(futs):
            p = futs[f]; c6_answers[p["pid"]] = f.result()
    c6_score = sum(score(extract_answer(c6_answers[p["pid"]]), p["gold"], p.get("tol"))
                    for p in test)

    # Print results
    n = len(test)
    print(f"\n=== End-to-end ablation on n={n} test problems ===")
    print(f"{'Cond':32s} {'accuracy':>10s} {'pp':>8s}")
    print("-" * 55)
    print(f"{'C0 baseline (no prior)':32s} {c0_score/n:>10.3f} {100*c0_score/n:>7.1f}%")
    print(f"{'C1 random-pick prior':32s} {c1_score_fresh/n:>10.3f} {100*c1_score_fresh/n:>7.1f}%")
    print(f"{'C2 best-single-prior ('+best_single+')':32s} {c2_score/n:>10.3f} {100*c2_score/n:>7.1f}%")
    print(f"{'C3 Stage-1 (LLM scheduler)':32s} {c3_score/n:>10.3f} {100*c3_score/n:>7.1f}%")
    print(f"{'C4 Stage-0.5 (world-model argmax)':32s} {c4_score/n:>10.3f} {100*c4_score/n:>7.1f}%")
    print(f"{'C5 Stage-1+2 (exp-aug scheduler)':32s} {c5_score/n:>10.3f} {100*c5_score/n:>7.1f}%")
    print(f"{'C6 FULL STACK':32s} {c6_score/n:>10.3f} {100*c6_score/n:>7.1f}%")
    print(f"\n  Per-stage incremental contribution:")
    print(f"    C0->C1 (any prior): {(c1_score_fresh-c0_score)/n:+.3f}")
    print(f"    C1->C2 (right prior on average): {(c2_score-c1_score_fresh)/n:+.3f}")
    print(f"    C2->C3 (per-problem scheduler): {(c3_score-c2_score)/n:+.3f}")
    print(f"    C3->C5 (+experience): {(c5_score-c3_score)/n:+.3f}")
    print(f"    C5->C6 (+world model): {(c6_score-c5_score)/n:+.3f}")
    print(f"\n    Total stack contribution (C0->C6): {(c6_score-c0_score)/n:+.3f}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_test": n,
           "scores": {
               "C0_baseline": c0_score / n,
               "C1_random_pick": c1_score_fresh / n,
               "C2_best_single": c2_score / n,
               "C3_stage1_only": c3_score / n,
               "C4_stage05_only": c4_score / n,
               "C5_stage12": c5_score / n,
               "C6_full_stack": c6_score / n,
           },
           "deltas": {
               "C0_to_C1_any_prior": (c1_score_fresh - c0_score) / n,
               "C1_to_C2_right_prior_avg": (c2_score - c1_score_fresh) / n,
               "C2_to_C3_scheduler": (c3_score - c2_score) / n,
               "C3_to_C5_experience": (c5_score - c3_score) / n,
               "C5_to_C6_worldmodel": (c6_score - c5_score) / n,
               "C0_to_C6_total": (c6_score - c0_score) / n,
           },
           "best_single_prior": best_single,
           "test_acc_per_fixed_prior": test_acc_per_prior,
           "c1_picks": c1_picks, "c5_picks": c5_picks, "c6_picks": c6_picks}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
