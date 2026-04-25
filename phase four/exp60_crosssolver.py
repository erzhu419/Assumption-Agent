"""Exp 60 — Cross-solver validation of the Stage 1 scheduler.

Closes the standing concern that the agent stack has only been
validated on gemini-3-flash. This experiment runs Exp 56's exact
protocol (60 problems × 4 families × 4 priors + LLM scheduler) on
two additional solver families: claude-haiku-4.5 and gpt-5.4-mini.

If the LLM scheduler beats fixed priors AND beats baseline on EACH
of the three solvers (gemini, claude-haiku, gpt-mini), we have
cross-solver evidence that Stage 1's value is not a gemini-specific
artefact. We re-use Exp 56's scheduler picks (same problems → same
picks, since the scheduler is gemini-prompted and we want to test
solver generalisation, not scheduler generalisation).

Conditions per solver (60 problems each):
  baseline (no prior) + 4 fixed priors + scheduler-picked + oracle
  = 7 conditions × 60 problems = 420 calls per solver
  × 2 new solvers = 840 calls + 60 reused scheduler picks = ~840
  cheap-tier.

Cost: ~$10-20.
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
OUT_LOG = AUTO / "exp60_crosssolver_log.json"

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

def solve(client, problem, prior_name):
    p = PRIORS[prior_name]
    approach = f"Use this strategy: {p}" if p else "Use any approach you think appropriate."
    try:
        r = client.generate(SOLVE_PROMPT.format(problem=problem, approach=approach),
                             max_tokens=600, temperature=0.0)
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"

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


def run_solver(solver_name, solver_client, all_problems, sched_picks):
    """Run all 7 conditions on one solver. Returns per-condition per-problem scores."""
    print(f"\n=== Running on solver: {solver_name} ===")
    conditions = ["baseline", "decompose", "restate", "estimate", "constraints"]
    answers = {c: {} for c in conditions}
    answers["scheduler"] = {}; answers["oracle"] = {}

    # Fixed conditions: 5 × 60 = 300 calls
    print(f"[1/3] Fixed conditions: 5 x 60 = 300 calls...")
    tasks = [(c, p) for c in conditions for p in all_problems]
    def run_fixed(c, p):
        prior = "none" if c == "baseline" else c
        return c, p["pid"], solve(solver_client, p["prompt"], prior)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(run_fixed, c, p) for c, p in tasks]
        for f in as_completed(futs):
            c, pid, ans = f.result(); answers[c][pid] = ans; done += 1
            if done % 60 == 0: print(f"  fixed {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    # Scheduler-picked: reuse picks from Exp 56 (gemini scheduler)
    print(f"\n[2/3] Scheduler-picked solve (60 calls; reusing gemini scheduler picks)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver_client, p["prompt"], sched_picks[p["pid"]]): p
                  for p in all_problems}
        for f in as_completed(futs):
            p = futs[f]; answers["scheduler"][p["pid"]] = f.result()
    print(f"  scheduler-solve done ({time.time()-t0:.0f}s)")

    # Oracle
    print(f"\n[3/3] Oracle (60 calls)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver_client, p["prompt"], p["optimal_prior"]): p
                  for p in all_problems}
        for f in as_completed(futs):
            p = futs[f]; answers["oracle"][p["pid"]] = f.result()
    print(f"  oracle done ({time.time()-t0:.0f}s)")

    # Score
    fams = ["A", "B", "C", "D"]
    cond_scores = {c: {f: [] for f in fams} for c in conditions + ["scheduler", "oracle"]}
    for p in all_problems:
        f = p["family"][0]
        for c in conditions + ["scheduler", "oracle"]:
            ext = extract_answer(answers[c].get(p["pid"], ""))
            sc = score(ext, p["gold"], p.get("tol"))
            cond_scores[c][f].append(sc)

    print(f"\n--- Per-family accuracy on {solver_name} ---")
    print(f"{'Condition':14s} {'A_dec':>7s} {'B_res':>7s} {'C_est':>7s} {'D_con':>7s} {'overall':>9s}")
    print("-" * 60)
    summary = {}
    for c in conditions + ["scheduler", "oracle"]:
        accs = [sum(cond_scores[c][f]) / len(cond_scores[c][f]) for f in fams]
        n_correct = sum(sum(cond_scores[c][f]) for f in fams)
        overall = n_correct / 60
        print(f"{c:14s} {accs[0]:>7.3f} {accs[1]:>7.3f} {accs[2]:>7.3f} {accs[3]:>7.3f} {overall:>9.3f}")
        summary[c] = {"A_acc": accs[0], "B_acc": accs[1], "C_acc": accs[2],
                       "D_acc": accs[3], "overall": overall, "n_correct": n_correct}

    return summary, answers


def main():
    print(f"=== Exp 60: cross-solver validation of Stage 1 scheduler ===")
    e56 = json.loads((AUTO / "exp56_stage1_broader_log.json").read_text())
    pp = e56["per_problem"]

    # Reconstruct problems
    all_problems = []
    for pid, d in pp.items():
        all_problems.append({"pid": pid, "prompt": d["prompt"], "gold": d["gold"],
                             "family": d["family"],
                             "optimal_prior": {"A_decompose": "decompose",
                                                 "B_restate": "restate",
                                                 "C_estimate": "estimate",
                                                 "D_constraints": "constraints"}[d["family"]],
                             "tol": 10 if d["family"] == "C_estimate" else None})

    sched_picks = e56["scheduler_picks"]

    # Run on each non-gemini solver
    solver_results = {}
    solver_results["gemini"] = {c: e56["scores"][c] for c in
                                  ["baseline", "decompose", "restate", "estimate",
                                   "constraints", "scheduler", "oracle"]}
    print(f"\n[gemini results from Exp 56:]")
    for c in ["baseline", "decompose", "restate", "estimate", "constraints", "scheduler", "oracle"]:
        s = solver_results["gemini"][c]
        accs = [s["A_acc"], s["B_acc"], s["C_acc"], s["D_acc"]]
        overall = e56["scores_overall"][c]
        print(f"  {c:14s} A={accs[0]:.3f} B={accs[1]:.3f} C={accs[2]:.3f} "
              f"D={accs[3]:.3f}  overall={overall:.3f}")

    for solver_name in ["claude_haiku", "gpt_mini"]:
        try:
            client = cheap(solver_name)
            print(f"\n  Solver: {client.model}")
        except Exception as e:
            print(f"  Could not initialize {solver_name}: {e}")
            continue
        summary, _ = run_solver(solver_name, client, all_problems, sched_picks)
        solver_results[solver_name] = summary

    # Final cross-solver comparison
    print(f"\n\n========================================================")
    print(f"=== Cross-solver summary (overall accuracy) ===")
    print(f"========================================================")
    conds = ["baseline", "decompose", "restate", "estimate", "constraints",
             "scheduler", "oracle"]
    print(f"{'Solver':14s} " + " ".join(f"{c:>10s}" for c in conds))
    print("-" * 95)
    for solver_name in ["gemini", "claude_haiku", "gpt_mini"]:
        if solver_name not in solver_results: continue
        if solver_name == "gemini":
            row = [e56["scores_overall"][c] for c in conds]
        else:
            row = [solver_results[solver_name][c]["overall"] for c in conds]
        print(f"{solver_name:14s} " + " ".join(f"{v:>10.3f}" for v in row))

    print(f"\n=== Scheduler vs best fixed (per solver) ===")
    for solver_name in solver_results:
        if solver_name == "gemini":
            sched = e56["scores_overall"]["scheduler"]
            fixed = max(e56["scores_overall"][c] for c in
                          ["decompose", "restate", "estimate", "constraints"])
        else:
            sched = solver_results[solver_name]["scheduler"]["overall"]
            fixed = max(solver_results[solver_name][c]["overall"] for c in
                          ["decompose", "restate", "estimate", "constraints"])
        print(f"  {solver_name:14s} scheduler={sched:.3f}  best_fixed={fixed:.3f}  "
              f"delta={sched - fixed:+.3f}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_problems": 60,
           "solvers_tested": list(solver_results.keys()),
           "results_per_solver": solver_results,
           "gemini_overall_acc_per_condition": e56["scores_overall"]}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
