"""Exp 58 — Stage 2 experience-feedback validation.

The reviewer's standing concern about Stage 2 is that ``no Stage 2
experience feedback is implemented or empirically tested.'' This
experiment shows that an experience-augmented scheduler outperforms
a naive scheduler on held-out problems, by feeding the scheduler
in-context examples of (problem, prior, outcome) triples from a
training split.

Design:
  Split Exp 56's 60 problems into TRAINING (40, 10 per family) and
  TEST (20, 5 per family). Both splits stratified by family.

  Naive scheduler (Exp 56's): given a test problem, pick a prior
    based only on the prompt.
  Experience-augmented scheduler: given a test problem, also
    receives the 40 (problem, picked_prior, outcome) triples from
    the training split as in-context retrieval. Pick a prior.

  Compare:
    - Pick accuracy: how often does the scheduler pick the optimal
      prior on the test set?
    - Solver accuracy: end-to-end, how often is the answer correct
      after applying the scheduler's pick?

If experience-augmented > naive on the test set, Stage 2 is
empirically validated at this minimal scale.

Cost: 20 test problems × 2 schedulers × (1 schedule + 1 solve) =
80 calls. ~$1.
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
OUT_LOG = AUTO / "exp58_stage2_experience_log.json"

PRIORS = {
    "decompose": "Before answering, decompose into atomic substeps.",
    "restate":   "Before answering, RE-READ the question; check if the obvious interpretation is a trap.",
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

NAIVE_SCHED_PROMPT = """You are a strategy selector. Given a problem, choose ONE
prior from: {{decompose, restate, estimate, constraints, none}}.

## Problem
{problem}

## Output (JSON)
{{"choice": "decompose"|"restate"|"estimate"|"constraints"|"none", "reason": "1 sentence"}}
"""

EXP_SCHED_PROMPT = """You are a strategy selector. Given a problem and PAST EXAMPLES
of (problem, picked_prior, was_correct) triples, learn from the examples and pick
the prior most likely to succeed on the new problem.

Choose ONE prior from: {{decompose, restate, estimate, constraints, none}}.

## Past examples (training experiences)
{examples}

## New problem
{problem}

## Output (JSON only)
{{"choice": "...", "reason": "1 sentence describing what pattern in past examples informs the choice"}}
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

def naive_schedule(client, problem):
    try:
        r = client.generate(NAIVE_SCHED_PROMPT.format(problem=problem),
                             max_tokens=200, temperature=0.0)
        m = re.search(r'"choice"\s*:\s*"(decompose|restate|estimate|constraints|none)"', r["text"])
        if m: return m.group(1)
    except Exception: pass
    return "none"

def exp_schedule(client, problem, training_triples, k=15):
    """Experience-augmented: provide k random training triples as in-context examples."""
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
    print(f"=== Exp 58: Stage 2 experience-feedback validation ===")
    e56 = json.loads((AUTO / "exp56_stage1_broader_log.json").read_text())
    pp = e56["per_problem"]

    # Reconstruct full problem objects
    all_problems = []
    for pid, d in pp.items():
        all_problems.append({
            "pid": pid, "prompt": d["prompt"], "gold": d["gold"],
            "family": d["family"],
            "optimal_prior": {"A_decompose": "decompose",
                                "B_restate": "restate",
                                "C_estimate": "estimate",
                                "D_constraints": "constraints"}[d["family"]],
            "tol": 10 if d["family"] == "C_estimate" else None,
        })

    # Stratified train/test split: 10 train + 5 test per family
    rng = random.Random(2026)
    fams = ["A_decompose", "B_restate", "C_estimate", "D_constraints"]
    train, test = [], []
    for f in fams:
        fam_p = [p for p in all_problems if p["family"] == f]
        rng.shuffle(fam_p)
        train += fam_p[:10]
        test += fam_p[10:15]
    print(f"  Train: {len(train)} problems (10 per family)")
    print(f"  Test:  {len(test)} problems (5 per family)\n")

    # Use Exp 56's data to build training triples: for each TRAIN
    # problem, take the answer/score from Exp 56's "scheduler" condition
    # — i.e. the naive scheduler's pick + outcome.
    # We already have these from per_problem in Exp 56.
    training_triples = []
    for p in train:
        d = pp[p["pid"]]
        # Use the "scheduler_score" and inferred pick
        # Exp 56 stored scheduler_picks separately; reconstruct prior used
        sched_pick = e56["scheduler_picks"].get(p["pid"], "none")
        sched_score = d.get("scheduler_score", 0)
        training_triples.append({
            "problem": p["prompt"], "prior": sched_pick,
            "outcome": int(sched_score),
            "family": p["family"]})

    # Verify training data has signal
    n_train_correct = sum(t["outcome"] for t in training_triples)
    print(f"  Training data: {n_train_correct}/{len(training_triples)} naive-scheduler "
          f"answers were correct ({n_train_correct/len(training_triples):.1%})\n")

    client = cheap("gemini")

    # Test condition 1: Naive scheduler (replicate Exp 56's logic on test)
    print(f"[1/2] Naive scheduler on {len(test)} test problems...")
    naive_picks, naive_answers = {}, {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(naive_schedule, client, p["prompt"]): p for p in test}
        for f in as_completed(futs):
            p = futs[f]; naive_picks[p["pid"]] = f.result()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, client, p["prompt"], naive_picks[p["pid"]]): p for p in test}
        for f in as_completed(futs):
            p = futs[f]; naive_answers[p["pid"]] = f.result()
    print(f"  Naive done ({time.time()-t0:.0f}s)")

    # Test condition 2: Experience-augmented scheduler
    print(f"\n[2/2] Experience-augmented scheduler on {len(test)} test problems...")
    exp_picks, exp_answers = {}, {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(exp_schedule, client, p["prompt"], training_triples): p for p in test}
        for f in as_completed(futs):
            p = futs[f]; exp_picks[p["pid"]] = f.result()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, client, p["prompt"], exp_picks[p["pid"]]): p for p in test}
        for f in as_completed(futs):
            p = futs[f]; exp_answers[p["pid"]] = f.result()
    print(f"  Exp-augmented done ({time.time()-t0:.0f}s)")

    # Score
    print(f"\n=== Test-set comparison ===")
    print(f"{'Cond':22s} {'pick_acc':>10s} {'solver_acc':>12s}")
    print("-" * 50)
    naive_pick_correct = naive_solver_correct = 0
    exp_pick_correct = exp_solver_correct = 0
    per_problem = {}
    for p in test:
        n_pick = naive_picks[p["pid"]]; e_pick = exp_picks[p["pid"]]
        n_ans = extract_answer(naive_answers[p["pid"]])
        e_ans = extract_answer(exp_answers[p["pid"]])
        n_score = score(n_ans, p["gold"], p.get("tol"))
        e_score = score(e_ans, p["gold"], p.get("tol"))
        if n_pick == p["optimal_prior"]: naive_pick_correct += 1
        if e_pick == p["optimal_prior"]: exp_pick_correct += 1
        naive_solver_correct += n_score
        exp_solver_correct += e_score
        per_problem[p["pid"]] = {"family": p["family"], "optimal": p["optimal_prior"],
                                    "naive_pick": n_pick, "naive_score": n_score,
                                    "exp_pick": e_pick, "exp_score": e_score}
    n_test = len(test)
    print(f"{'naive scheduler':22s} {naive_pick_correct/n_test:>10.3f} {naive_solver_correct/n_test:>12.3f}")
    print(f"{'experience-augmented':22s} {exp_pick_correct/n_test:>10.3f} {exp_solver_correct/n_test:>12.3f}")
    delta_pick = (exp_pick_correct - naive_pick_correct) / n_test
    delta_solver = (exp_solver_correct - naive_solver_correct) / n_test
    print(f"\n  Delta (exp - naive): pick {delta_pick:+.3f}, solver {delta_solver:+.3f}")
    print(f"  ({'experience helps' if delta_pick > 0 else 'no improvement from experience'})")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_train": len(train), "n_test": len(test),
           "naive_pick_accuracy": naive_pick_correct / n_test,
           "exp_pick_accuracy": exp_pick_correct / n_test,
           "naive_solver_accuracy": naive_solver_correct / n_test,
           "exp_solver_accuracy": exp_solver_correct / n_test,
           "delta_pick": delta_pick, "delta_solver": delta_solver,
           "per_problem": per_problem,
           "training_triples_used": training_triples,
           "naive_picks": naive_picks, "exp_picks": exp_picks}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
