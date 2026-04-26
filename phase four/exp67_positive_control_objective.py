"""Exp 67 — Positive control with OBJECTIVE grading (v2: targeted).

Closes the most serious open reviewer concern (R30/R31/R32):
the previous positive controls (Exp 44/46) all failed (wr <= 0.25),
suggesting the gate may be structurally anti-additive. But those
experiments judged on Chinese open-ended problems, where the
same-family judge has a strong "no-prior is more concise" bias
regardless of content.

This experiment isolates the question by switching to a domain where
correctness is OBJECTIVELY VERIFIABLE.

DESIGN
======
Problem set: the 16 problems from Exp 62 (Bayesian/probabilistic +
multi-step arithmetic + logic + fermi) on which the BASELINE solve
(no prior) was *demonstrably wrong* against the gold answer. By
construction these are problems where there is HEADROOM for a prior
to help.

Three conditions per problem:

  BASE      : prior = "" (whatever the model wants).

  POSITIVE  : a strongly-algorithmic prior:
              "Step 1 - re-read the problem; identify what is asked.
               Step 2 - list every given constraint/quantity explicitly.
               Step 3 - identify the problem type and the appropriate
                        systematic procedure.
               Step 4 - apply the procedure; show ALL work.
               Step 5 - verify by substituting your answer back into
                        the constraints; if any constraint is violated,
                        revise.
               Step 6 - state the final answer in the requested format."
              This is the strongest generic methodological prior I can
              construct without using problem-specific code.

  NEGATIVE  : a deliberately-harmful prior that suppresses reasoning:
              "Skip all reasoning. Pattern-match to the most familiar
               similar problem you have seen and write only the final
               numerical answer on a single line. Do NOT show work,
               check, or revise."
              This forces no-CoT on problems that demand CoT, so it
              should objectively hurt accuracy.

OUTCOME MAP
===========

We compute:
  - Δacc_pos = obj_acc(POSITIVE) - obj_acc(BASE)
  - Δacc_neg = obj_acc(NEGATIVE) - obj_acc(BASE)
  - wr_pos_inner / wr_pos_L1 = same-family / cross-family A/B verdict
                                for POSITIVE-vs-BASE
  - wr_neg_inner / wr_neg_L1 = same-family / cross-family A/B verdict
                                for NEGATIVE-vs-BASE

Outcomes:
  A — audit stack works: Δacc_pos > 0 AND wr_pos_inner >= 0.55 AND
        wr_pos_L1 >= 0.55 AND wr_neg_inner < 0.50 AND wr_neg_L1 < 0.50.
        The audit stack IS sensitive to known-positive (and
        known-negative) insertions on objectively-graded tasks. The
        original 12-candidate null is content-driven, not
        gate-architectural.

  B — same-family bias only: Δacc_pos > 0 AND wr_pos_inner < 0.55 BUT
        wr_pos_L1 >= 0.55. -> "L1 cross-family is a NECESSARY audit
        layer; same-family alone misses real positive interventions."

  C — gate is structurally anti-additive: Δacc_pos > 0 BUT
        wr_pos_inner < 0.55 AND wr_pos_L1 < 0.55. -> "Even objectively
        positive interventions are rejected by the entire gate+L1
        stack. Same-family A/B at n=50 is fundamentally unfit for
        validation in any retrieval-level self-improvement loop."

  D — positive control failed: Δacc_pos <= 0. The strongly-algorithmic
        prior did not measurably help even on baseline-wrong problems.
        Two interpretations: (i) the solver model commits to its first
        answer regardless of prior strength on these tasks
        ("stubborn at temp=0"); (ii) the trap-types in our problem
        set are not the kind that a generic algorithmic prior can
        rescue. Either way, the question of audit-stack sensitivity
        cannot be answered by this protocol; report honestly.

Cost: 16 x 3 = 48 solves + 16 x 3 pairs x 2 judges = 96 judgments
= ~144 calls total. ~$1.5, ~15 min.
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

# Reuse the EXACT same problem set and scoring infra from exp62
import importlib.util
_spec = importlib.util.spec_from_file_location("exp62", PROJECT / "phase four" / "exp62_harder_tasks.py")
_exp62 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exp62)
ALL_PROBLEMS = _exp62.ALL_PROBLEMS
SOLVE = _exp62.SOLVE
extract = _exp62.extract
score_n = _exp62.score_n

PARALLEL = 6
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp67_positive_control_objective_log.json"

# Strongly-algorithmic prior. Designed to help on hard reasoning problems.
POSITIVE_PRIOR = """Solve this problem with the following systematic procedure:

Step 1: Re-read the problem. State in your own words EXACTLY what is being asked.
Step 2: List every constraint and quantity given in the problem, explicitly and separately.
Step 3: Identify the problem type (probability with conditioning / multi-step arithmetic / logical deduction / counting / etc.) and the appropriate systematic procedure (Bayes' rule with explicit conditional / step-by-step accounting / case enumeration / etc.).
Step 4: Apply the procedure. Show every intermediate computation explicitly. Do not skip steps.
Step 5: Verify your answer by substituting it back into the problem's constraints. If any constraint is violated, identify the error and revise.
Step 6: Write the final answer in the format the problem requested."""

# Reasoning-suppressing prior. Designed to hurt on hard reasoning problems.
NEGATIVE_PRIOR = """Skip all reasoning. Do NOT enumerate cases, apply formulas, or check your work. Pattern-match this problem to the most familiar similar problem you have seen and write only the final answer on a single line. Do not show work."""

CONDITIONS = {
    "BASE":     "",
    "POSITIVE": POSITIVE_PRIOR,
    "NEGATIVE": NEGATIVE_PRIOR,
}

# Pick the 16 problems where Exp 62's baseline got it wrong.
def select_problems():
    e62 = json.loads((PROJECT / "phase four" / "autonomous"
                       / "exp62_harder_tasks_log.json").read_text())
    pp = e62["per_problem"]
    wrong = [pid for pid, info in pp.items() if info.get("baseline_score") == 0]
    pid_to_problem = {p["pid"]: p for p in ALL_PROBLEMS}
    out = [pid_to_problem[pid] for pid in wrong if pid in pid_to_problem]
    return out


def solve(client, problem_text, prior):
    appr = f"Use this strategy: {prior}" if prior else "Use any approach you think appropriate."
    try:
        r = client.generate(SOLVE.format(problem=problem_text, approach=appr),
                             max_tokens=800, temperature=0.0)
        return r["text"].strip()
    except Exception as e:
        return f"[err: {e}]"


def score_objective(answer_text, gold, tol=None):
    ext = extract(answer_text)
    if isinstance(gold, str):
        return 1 if gold.lower().strip() in ext.lower() else 0
    return score_n(ext, gold, tol)


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
    return {
        "wins_a": wins_a, "wins_b": wins_b, "ties": ties,
        "n_eff": n_eff, "wr_a": wr_a, "per_pid": per_pid,
    }


def main():
    print(f"=== Exp 67 v2: positive control with objective grading (targeted) ===", flush=True)
    problems = select_problems()
    print(f"  selected {len(problems)} problems where Exp 62 baseline was wrong", flush=True)
    fams = defaultdict(int)
    for p in problems: fams[p["family"]] += 1
    print(f"  family breakdown: {dict(fams)}", flush=True)

    solver = cheap("gemini")
    judge_g = cheap("gemini")
    judge_h = cheap("claude_haiku")

    # ---- Stage 1: Generate three answers per problem ------------------
    print(f"\n[1/3] Generating answers (3 conditions x {len(problems)} = "
          f"{3 * len(problems)} solves)...", flush=True)
    t0 = time.time()
    answers = {c: {} for c in CONDITIONS}

    def solve_task(cond, p):
        return cond, p["pid"], solve(solver, p["prompt"], CONDITIONS[cond])

    tasks = [(c, p) for c in CONDITIONS for p in problems]
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
    for cond in CONDITIONS:
        per_fam = defaultdict(lambda: [0, 0])
        for p in problems:
            ans = answers[cond].get(p["pid"], "")
            sc = score_objective(ans, p["gold"], p.get("tol"))
            per_fam[p["family"]][0] += sc
            per_fam[p["family"]][1] += 1
        n_correct = sum(v[0] for v in per_fam.values())
        n_total = sum(v[1] for v in per_fam.values())
        obj[cond] = {
            "overall": {"correct": n_correct, "total": n_total,
                          "acc": n_correct / n_total if n_total else 0},
            "per_family": {f: {"correct": v[0], "total": v[1],
                                 "acc": v[0] / v[1]} for f, v in per_fam.items()},
        }
        print(f"  {cond:10s}: {n_correct}/{n_total} = {n_correct/n_total:.1%}", flush=True)
        for fam, v in per_fam.items():
            print(f"    {fam:18s}: {v[0]}/{v[1]} = {v[0]/v[1]:.1%}", flush=True)

    delta_pos = obj["POSITIVE"]["overall"]["acc"] - obj["BASE"]["overall"]["acc"]
    delta_neg = obj["NEGATIVE"]["overall"]["acc"] - obj["BASE"]["overall"]["acc"]
    print(f"\n  Δacc(POSITIVE - BASE) = {delta_pos:+.1%}", flush=True)
    print(f"  Δacc(NEGATIVE - BASE) = {delta_neg:+.1%}", flush=True)

    # ---- Stage 3: Pairwise judging ------------------------------------
    print(f"\n[3/3] Pairwise A/B judging (3 pairs x 2 judges x "
          f"{len(problems)} = {6*len(problems)} judge calls)...", flush=True)
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
            r = run_pairwise(jclient, answers[la], answers[lb], la, lb, problems)
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

    if delta_pos <= 0:
        outcome = "D — positive control failed (algorithmic prior didn't help objectively)"
    elif pos_inner >= 0.55 and pos_L1 >= 0.55 and neg_inner < 0.50 and neg_L1 < 0.50:
        outcome = "A — audit stack IS sensitive to known-positive insertions"
    elif pos_inner < 0.55 and pos_L1 >= 0.55:
        outcome = "B — same-family judge has style bias; L1 corrects"
    elif pos_inner < 0.55 and pos_L1 < 0.55:
        outcome = "C — gate is STRUCTURALLY ANTI-ADDITIVE even on objectively-positive insertions"
    else:
        outcome = "mixed — see numerical detail"

    print(f"\n=== OUTCOME ===", flush=True)
    print(f"  Δacc_pos = {delta_pos:+.1%}, Δacc_neg = {delta_neg:+.1%}", flush=True)
    print(f"  pos_inner_wr = {pos_inner:.3f}, pos_L1_wr = {pos_L1:.3f}", flush=True)
    print(f"  neg_inner_wr = {neg_inner:.3f}, neg_L1_wr = {neg_L1:.3f}", flush=True)
    print(f"  Verdict: {outcome}", flush=True)

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "n_problems": len(problems),
        "problem_pids": [p["pid"] for p in problems],
        "selection_criterion": "Exp 62 baseline_score == 0 (model demonstrably wrong without prior)",
        "conditions": {k: v for k, v in CONDITIONS.items()},
        "objective_accuracy": obj,
        "delta_acc_pos_minus_base": delta_pos,
        "delta_acc_neg_minus_base": delta_neg,
        "pairwise": pairwise,
        "outcome_verdict": outcome,
        "answers": answers,
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}", flush=True)


if __name__ == "__main__":
    main()
