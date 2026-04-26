"""Exp 69 — Positive control re-run with a weaker solver
(gpt-5.4-mini) to widen the accuracy headroom.

Closes the residual reviewer concern (R33) that Exp 68's
Δacc=+6.7% was small (1/15 rescued) because gemini-3-flash's
baseline on the 15 Bayesian counter-intuitive problems was already
14/15, leaving no room for the prior to demonstrate substantial
accuracy improvement. The reviewer worried that wr=0.846 might
reflect preference for rigour/explicitness rather than for
correctness.

Same 15 problems, same POSITIVE/NEGATIVE priors, same judges as
Exp 68. The only change: the SOLVER is gpt-5.4-mini instead of
gemini-3-flash. We expect gpt-5.4-mini to be weaker on Bayesian
counter-intuitive problems, giving a wider accuracy headroom for
the task-specific Bayesian template to demonstrate Δacc.

DESIGN
======
Two changes vs Exp 67 v2:

(1) PROBLEM SELECTION. We hand-curate 15 Bayesian counter-intuitive
    problems --- problem types where strong LLMs are known to commit
    typical cognitive biases (base-rate neglect, conjunction
    interpretation, conditional probability via enumeration). These
    are the kinds of problems for which an explicit Bayesian
    procedure measurably helps because the model's natural
    chain-of-thought tends to short-circuit to a heuristic answer.

(2) PRIOR SPECIFICITY. Instead of a generic algorithmic prior
    ("list constraints, apply procedure, verify"), we use a
    task-specific PROCEDURAL TEMPLATE for Bayesian problems with
    two worked examples. The prior teaches the solver:
       - enumerate the full sample space when discrete
       - apply P(A | B) = |A and B| / |B| explicitly
       - verify by substituting back
    This is closer to what an actual methodological prior the loop
    would commit looks like (a procedure, not a maxim).

Same gate machinery as Exp 67: judge_pair (gemini-3-flash) for
inner gate + claude-haiku-4.5 for L1 cross-family audit.

OUTCOME MAP
===========
Goal: Δacc(POSITIVE - BASE) >= +20pp AND wr_pos_inner >= 0.60
AND wr_pos_L1 >= 0.55. If achieved, we have a true positive
control: a known-useful intervention that the gate accepts.
This would close the open question on audit-stack sensitivity.

If Δacc moves but gate doesn't accept: gate is structurally
anti-additive even on demonstrably-useful interventions
(this is the strong vision-paper result).

If Δacc doesn't move: the targeted prior is also too weak;
we need a stronger intervention (e.g. explicit problem-specific
formula injection per problem) — which would itself be too
intrusive to be a "wisdom".

Cost: 15 x 3 = 45 solves + 15 x 3 pairs x 2 judges = 90 judgments
= ~135 calls. ~$2, ~15 min.
"""
import json, os, random, re, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))
sys.path.insert(0, str(PROJECT / "phase one" / "scripts" / "validation"))
sys.path.insert(0, str(PROJECT / "phase four"))

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

PARALLEL = 6
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp69_positive_control_weaker_solver_log.json"

# 15 Bayesian counter-intuitive problems. Numeric gold + tolerance.
PROBLEMS = [
    # Group A: classic Monty Hall variants
    {"pid": "B01", "prompt": "There are 3 doors. Behind one is a car, behind the other two are goats. You pick door 1. Monty (who knows what's behind every door) opens door 3 to reveal a goat. He offers you to switch to door 2. What is the probability of winning the car if you SWITCH? Express as a decimal to 3 places.", "gold": 0.667, "tol": 1.05},
    {"pid": "B02", "prompt": "Same Monty Hall setup but with 5 doors and 1 car. You pick door 1. Monty opens 3 of the remaining 4 doors all revealing goats. You switch to the last unopened door. P(car)?  Decimal to 3 places.", "gold": 0.8, "tol": 1.05},

    # Group B: two-children / multi-event
    {"pid": "B03", "prompt": "Two children. At least one is a boy. What is the probability both are boys? Decimal.", "gold": 0.333, "tol": 1.05},
    {"pid": "B04", "prompt": "Two children. The OLDER one is a boy. What is the probability both are boys? Decimal.", "gold": 0.5, "tol": 1.05},
    {"pid": "B05", "prompt": "Three children. At least one is a boy. P(all three are boys)? Decimal to 3 places.", "gold": 0.143, "tol": 1.10},

    # Group C: dice/card conditional
    {"pid": "B06", "prompt": "Roll 2 fair 6-sided dice. Given the sum is 7, what is the probability one of the dice shows 4? Decimal.", "gold": 0.333, "tol": 1.05},
    {"pid": "B07", "prompt": "Three cards: card 1 has both sides red, card 2 has both sides blue, card 3 has one red side and one blue side. You pick a card at random and look at one side: it is red. What is the probability the OTHER side is also red? Decimal.", "gold": 0.667, "tol": 1.05},
    {"pid": "B08", "prompt": "Two coins. Coin A is fair. Coin B has heads on both sides. You pick a coin at random and flip it: heads. P(picked coin A)? Decimal.", "gold": 0.333, "tol": 1.05},

    # Group D: base-rate disease
    {"pid": "B09", "prompt": "A disease affects 1% of a population. A test is 99% sensitive (P(positive | disease) = 0.99) and 99% specific (P(negative | no disease) = 0.99). Given a positive test, what is the probability the person actually has the disease? Decimal to 3 places.", "gold": 0.5, "tol": 1.05},
    {"pid": "B10", "prompt": "Disease prevalence 0.1%. Test sensitivity 99%, specificity 95%. P(disease | positive)? Decimal to 4 places.", "gold": 0.0194, "tol": 1.20},
    {"pid": "B11", "prompt": "5% of patients have a condition. Test correctly catches 90% of true positives (sensitivity 0.9), but flags 10% of healthy patients (false-positive rate 0.1). P(condition | positive test)? Decimal to 3 places.", "gold": 0.321, "tol": 1.10},

    # Group E: drug-test / signal-detection
    {"pid": "B12", "prompt": "1 in 1000 athletes uses a banned drug. The drug test has 1% false-positive rate and 0% false-negative rate. An athlete tests positive. P(actually used the drug)? Decimal to 3 places.", "gold": 0.091, "tol": 1.15},
    {"pid": "B13", "prompt": "Boxes A and B. A contains 4 white and 6 black balls. B contains 7 white and 3 black. You pick a box at random and draw a ball: it is white. P(it came from B)? Decimal to 3 places.", "gold": 0.636, "tol": 1.10},

    # Group F: counterintuitive sample-space
    {"pid": "B14", "prompt": "A bag contains 1 red ball and 99 blue balls. You draw 5 without replacement. P(the red ball is among the 5 drawn)? Decimal to 3 places.", "gold": 0.05, "tol": 1.10},
    {"pid": "B15", "prompt": "You ask 100 people their birth month. P(at least two share a birth month) assuming uniform? Decimal to 3 places.", "gold": 1.0, "tol": 1.001},
]

# Strongly task-specific prior: Bayesian template + 2 worked examples.
POSITIVE_PRIOR = """For probability problems involving conditional events ("given X, what is P(Y)?"), follow this procedure:

1. ENUMERATE: list every possible outcome of the underlying random process. For example, "two children" gives 4 outcomes: BB, BG, GB, GG (each equally likely a priori).

2. CONDITION: identify the conditioning event. Restrict the sample space to outcomes consistent with what is given.

3. COMPUTE: P(target | condition) = (# outcomes satisfying both target AND condition) / (# outcomes satisfying condition).

WORKED EXAMPLE 1 — Two children, at least one is a boy. P(both boys)?
  Sample space: BB, BG, GB, GG (4 equally likely).
  Condition "at least one boy" rules out GG; remaining: BB, BG, GB (3 outcomes).
  Target "both boys" = BB (1 outcome).
  P = 1/3 ≈ 0.333.

WORKED EXAMPLE 2 — Disease 1% prevalence; test 99% sensitive, 99% specific. P(disease | positive)?
  Out of 10000 people: 100 sick, 9900 healthy.
  Of 100 sick: 99 test positive (sensitivity 99%).
  Of 9900 healthy: 99 test positive (1% false-positive rate).
  Total positive tests: 99 + 99 = 198.
  P(sick | positive) = 99 / 198 = 0.5.

NOW SOLVE THE PROBLEM. Enumerate, condition, compute. Show every step explicitly."""

# Reasoning-suppressing prior (same as Exp 67's NEGATIVE).
NEGATIVE_PRIOR = """Skip all reasoning. Do NOT enumerate sample spaces, apply Bayes' rule, or check your work. Pattern-match this problem to the most familiar similar problem you have seen and write only the final numerical answer on a single line. Do not show work."""

CONDITIONS = {
    "BASE":     "",
    "POSITIVE": POSITIVE_PRIOR,
    "NEGATIVE": NEGATIVE_PRIOR,
}

SOLVE_PROMPT = """## Problem
{problem}

## Approach hint
{approach}

## Output
Reason step by step, then on the LAST LINE write exactly:
ANSWER: <numeric answer only>
"""


def solve(client, problem_text, prior):
    appr = f"Use this strategy: {prior}" if prior else "Use any approach you think appropriate."
    try:
        r = client.generate(SOLVE_PROMPT.format(problem=problem_text, approach=appr),
                             max_tokens=1200, temperature=0.0)
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"


def extract_numeric(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    raw = (m.group(1) if m else text[-200:]).strip()
    s = raw.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    nm = re.search(r'-?\d+(?:\.\d+)?(?:[eE]-?\d+)?', s)
    if not nm: return None
    try:
        return float(nm.group())
    except Exception:
        return None


def score(answer_text, gold, tol):
    val = extract_numeric(answer_text)
    if val is None:
        return 0
    if tol >= 1.001:
        # multiplicative tolerance: tol = 1.05 means within 5%
        if gold == 0:
            return 1 if abs(val) < 0.001 else 0
        rel = abs(val - gold) / abs(gold) if gold != 0 else abs(val)
        return 1 if rel <= (tol - 1.0) else 0
    # absolute tolerance
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
        if w == "tie":
            return "tie"
        return label_a if w == a_was else label_b
    except Exception:
        return "tie"


def run_pairwise(judge, ans_dict_a, ans_dict_b, label_a, label_b, problems):
    wins_a = wins_b = ties = 0
    per_pid = {}
    def one(p):
        pid = p["pid"]
        a = ans_dict_a.get(pid, "")
        b = ans_dict_b.get(pid, "")
        if not a or not b:
            return pid, "tie"
        seed = hash(pid + label_a + label_b) % (2**32)
        return pid, judge_with_side_randomize(judge, p["prompt"], a, b, label_a, label_b, seed)
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(one, p) for p in problems]
        for f in as_completed(futs):
            pid, w = f.result()
            per_pid[pid] = w
            if w == label_a: wins_a += 1
            elif w == label_b: wins_b += 1
            else: ties += 1
    _save_content_cache()
    n_eff = wins_a + wins_b
    wr_a = wins_a / n_eff if n_eff else 0.5
    return {"wins_a": wins_a, "wins_b": wins_b, "ties": ties,
            "n_eff": n_eff, "wr_a": wr_a, "per_pid": per_pid}


def main():
    print(f"=== Exp 68: TRUE positive control on Bayesian counter-intuitive problems ===", flush=True)
    print(f"  problems: {len(PROBLEMS)} hand-curated Bayesian / conditional probability", flush=True)

    solver = cheap("gpt_mini")             # SWAPPED: gpt-5.4-mini, weaker than gemini on these
    judge_g = cheap("gemini")               # judges unchanged: same as Exp 68
    judge_h = cheap("claude_haiku")

    # ---- Stage 1: Generate three answers per problem ------------------
    print(f"\n[1/3] Generating answers (3 conditions x {len(PROBLEMS)} = "
          f"{3 * len(PROBLEMS)} solves)...", flush=True)
    t0 = time.time()
    answers = {c: {} for c in CONDITIONS}

    def solve_task(cond, p):
        return cond, p["pid"], solve(solver, p["prompt"], CONDITIONS[cond])

    tasks = [(c, p) for c in CONDITIONS for p in PROBLEMS]
    done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(solve_task, c, p) for c, p in tasks]
        for f in as_completed(futs):
            c, pid, ans = f.result()
            answers[c][pid] = ans
            done += 1
            if done % 10 == 0:
                print(f"  solves {done}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)
    print(f"  solves done in {time.time()-t0:.0f}s", flush=True)

    # ---- Stage 2: Objective grading -----------------------------------
    print(f"\n[2/3] Objective grading vs gold...", flush=True)
    obj = {}
    per_pid_correct = {c: {} for c in CONDITIONS}
    for cond in CONDITIONS:
        n_correct = 0
        for p in PROBLEMS:
            ans = answers[cond].get(p["pid"], "")
            sc = score(ans, p["gold"], p.get("tol", 1.05))
            per_pid_correct[cond][p["pid"]] = sc
            n_correct += sc
        obj[cond] = {"correct": n_correct, "total": len(PROBLEMS),
                       "acc": n_correct / len(PROBLEMS)}
        print(f"  {cond:10s}: {n_correct}/{len(PROBLEMS)} = {n_correct/len(PROBLEMS):.1%}", flush=True)

    delta_pos = obj["POSITIVE"]["acc"] - obj["BASE"]["acc"]
    delta_neg = obj["NEGATIVE"]["acc"] - obj["BASE"]["acc"]
    print(f"\n  Δacc(POSITIVE - BASE) = {delta_pos:+.1%}", flush=True)
    print(f"  Δacc(NEGATIVE - BASE) = {delta_neg:+.1%}", flush=True)

    # Per-pid where POSITIVE rescued BASE
    rescued = [pid for pid in per_pid_correct["BASE"]
                if per_pid_correct["BASE"][pid] == 0
                and per_pid_correct["POSITIVE"][pid] == 1]
    print(f"  POSITIVE rescued BASE on: {len(rescued)} problems", flush=True)

    # ---- Stage 3: Pairwise judging ------------------------------------
    print(f"\n[3/3] Pairwise A/B judging (3 pairs x 2 judges x "
          f"{len(PROBLEMS)} = {6*len(PROBLEMS)} judge calls)...", flush=True)
    pairs = [
        ("POSITIVE_vs_BASE",  "POSITIVE", "BASE"),
        ("NEGATIVE_vs_BASE",  "NEGATIVE", "BASE"),
        ("POSITIVE_vs_NEGATIVE", "POSITIVE", "NEGATIVE"),
    ]
    judges = [("gemini_innerlike", judge_g), ("haiku_L1", judge_h)]
    pairwise = {}
    for name_pair, la, lb in pairs:
        pairwise[name_pair] = {}
        for jname, jclient in judges:
            t1 = time.time()
            r = run_pairwise(jclient, answers[la], answers[lb], la, lb, PROBLEMS)
            r["wr_a_label"] = la
            pairwise[name_pair][jname] = r
            print(f"  {name_pair:25s} | {jname:18s}: "
                  f"wr_{la}={r['wr_a']:.3f} ({r['wins_a']}-{r['wins_b']}-{r['ties']} t) "
                  f"({time.time()-t1:.0f}s)", flush=True)

    # ---- Outcome diagnosis --------------------------------------------
    pos_inner = pairwise["POSITIVE_vs_BASE"]["gemini_innerlike"]["wr_a"]
    pos_L1 = pairwise["POSITIVE_vs_BASE"]["haiku_L1"]["wr_a"]
    neg_inner = pairwise["NEGATIVE_vs_BASE"]["gemini_innerlike"]["wr_a"]
    neg_L1 = pairwise["NEGATIVE_vs_BASE"]["haiku_L1"]["wr_a"]

    if delta_pos < 0.05:
        outcome = "D — POSITIVE prior didn't move objective accuracy enough; not a true positive control"
    elif delta_pos >= 0.20 and pos_inner >= 0.60 and pos_L1 >= 0.55:
        outcome = "A — TRUE POSITIVE CONTROL ACHIEVED. Audit stack accepts an objectively-helpful intervention."
    elif delta_pos >= 0.20 and pos_inner < 0.60 and pos_L1 < 0.55:
        outcome = "C — gate is STRUCTURALLY ANTI-ADDITIVE: rejects intervention that demonstrably raises accuracy"
    elif delta_pos >= 0.20 and pos_inner < 0.60 and pos_L1 >= 0.55:
        outcome = "B — same-family judge has style bias against insertion; L1 cross-family corrects"
    else:
        outcome = f"mixed — Δacc={delta_pos:+.1%}, see numbers"

    print(f"\n=== OUTCOME ===", flush=True)
    print(f"  Δacc_pos = {delta_pos:+.1%}, Δacc_neg = {delta_neg:+.1%}", flush=True)
    print(f"  POSITIVE rescued BASE on {len(rescued)}/{len(PROBLEMS)} problems", flush=True)
    print(f"  pos_inner_wr = {pos_inner:.3f}, pos_L1_wr = {pos_L1:.3f}", flush=True)
    print(f"  neg_inner_wr = {neg_inner:.3f}, neg_L1_wr = {neg_L1:.3f}", flush=True)
    print(f"  Verdict: {outcome}", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_problems": len(PROBLEMS),
        "problem_pids": [p["pid"] for p in PROBLEMS],
        "task_description": "15 hand-curated Bayesian counter-intuitive problems",
        "conditions": CONDITIONS,
        "objective_accuracy": obj,
        "delta_acc_pos_minus_base": delta_pos,
        "delta_acc_neg_minus_base": delta_neg,
        "rescued_pids": rescued,
        "per_pid_correct": per_pid_correct,
        "pairwise": pairwise,
        "outcome_verdict": outcome,
        "answers": answers,
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}", flush=True)


if __name__ == "__main__":
    main()
