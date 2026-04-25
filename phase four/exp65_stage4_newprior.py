"""Exp 65 — Stage 4 minimal: agent generates a NEW prior given examples.

The most ambitious stage: when the existing prior library does not
fit a problem family well, the agent proposes a new prior. This
experiment tests the simplest possible Stage 4 instantiation.

Protocol (leave-one-family-out):
  1. Hold out one of Exp 62's 5 hard families (E/F/G/H/I).
  2. Show the agent the OTHER 4 families' (problem, optimal-prior) pairs.
  3. Ask the agent to propose a NEW prior tailored to the held-out family,
     given example problems from it (without revealing the optimal prior).
  4. Test: solve the held-out family with the proposed new prior;
     compare to baseline, to scheduler-from-existing-4, and to oracle.

If the proposed prior improves accuracy over baseline AND scheduler,
Stage 4 is empirically demonstrated at minimal scale. If it matches
or approaches oracle, the agent has correctly identified the
methodological structure of the new family.

We run this 5 times (one per held-out family) to average over
which family is held out.

Cost: 5 hold-outs × (1 prior-generation + 15 baseline + 15 new-prior
+ 15 oracle solver calls + 15 scheduler picks) = 5 × 61 = 305 calls
plus ~50 prior-generation calls. ~$5.
"""
import json, os, re, sys, time
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
OUT_LOG = AUTO / "exp65_stage4_newprior_log.json"

EXISTING_PRIORS = {
    "decompose": "Before answering, decompose into atomic substeps.",
    "restate": "Before answering, RE-READ the question carefully; check if the obvious interpretation is a trap.",
    "estimate": "Before answering, give an order-of-magnitude estimate; sanity-check final answer.",
    "constraints": "Before answering, enumerate explicit constraints; satisfy all simultaneously.",
}

GENERATE_PRIOR_PROMPT = """You are a methodology designer. Given examples of
problems and the methodological prior that helps solve them, propose a NEW
prior tailored to a new family of problems.

## Existing prior library (with example problems each helps)
{prior_examples}

## NEW problem family (a few examples, no prior given)
{new_problems}

## Your task
Look at the new problems. Identify what methodological pattern they share.
Propose a SHORT (one sentence) prior — a rule of thumb the solver should
apply BEFORE answering — that is targeted at this family.

The prior should be DIFFERENT from {existing_priors_list} but methodologically
useful. It should be one specific actionable instruction.

## Output (JSON only)
{{
  "name": "snake_case_name (one or two words)",
  "prior_text": "Before answering, ... (one actionable sentence)",
  "rationale": "1-2 sentences on why this prior fits the new family"
}}
"""

SOLVE_PROMPT = """## Problem
{problem}

## Approach hint
{approach}

## Output
Reason step by step in 1-3 sentences, then on the LAST LINE write exactly:
ANSWER: <your final answer>
"""

def solve(client, problem, prior_text):
    appr = f"Use this strategy: {prior_text}" if prior_text else "Use any approach you think appropriate."
    try:
        r = client.generate(SOLVE_PROMPT.format(problem=problem, approach=appr),
                             max_tokens=600, temperature=0.0)
        return r["text"].strip()
    except Exception as e: return f"[err: {e}]"

def generate_prior(client, prior_examples, new_problems, existing_names):
    try:
        r = client.generate(
            GENERATE_PRIOR_PROMPT.format(
                prior_examples=prior_examples,
                new_problems=new_problems,
                existing_priors_list=", ".join(existing_names)),
            max_tokens=400, temperature=0.3)
        text = r["text"].strip()
        # Try to parse JSON
        m = re.search(r'\{[^{}]*"prior_text"\s*:\s*"([^"]+)"[^{}]*\}', text, re.DOTALL)
        if m:
            prior_text = m.group(1)
            # Also extract name
            nm = re.search(r'"name"\s*:\s*"([^"]+)"', text)
            name = nm.group(1) if nm else "generated"
            return {"name": name, "prior_text": prior_text, "raw": text}
    except Exception as e:
        pass
    return {"name": "fallback", "prior_text": "", "raw": ""}

def extract(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()

def score_n(ext, gold, tol=None):
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
    return 1 if abs(ev - gv) < 0.05 else 0

def score_t(ext, gold):
    e = ext.lower().strip().rstrip(".")
    g = str(gold).lower().strip()
    if g in e: return 1
    g_w = set(re.findall(r'\w+', g))
    mean = {w for w in g_w if len(w) >= 3}
    if not mean: return 0
    e_w = set(re.findall(r'\w+', e))
    return 1 if len(mean & e_w) / len(mean) >= 0.5 else 0

def sc(ext, gold, tol=None):
    if isinstance(gold, (int, float)): return score_n(ext, gold, tol)
    return score_t(ext, gold)


def main():
    print(f"=== Exp 65: Stage 4 minimal — generate new prior given examples ===")
    e62_path = AUTO / "exp62_harder_tasks_log.json"
    if not e62_path.exists():
        print(f"ERROR: Exp 62 log not found.")
        return
    e62 = json.loads(e62_path.read_text())
    pp = e62["per_problem"]

    # Reconstruct hard problems
    fam_optimal = {"E_probabilistic": "constraints", "F_multistep": "decompose",
                    "G_wordtrap": "restate", "H_logic": "constraints",
                    "I_fermi": "estimate"}
    # We'll need original prompts. Read them from Exp 62's source code
    # alternatively store gold + family per pid. Reconstruct from Exp 62's
    # source via importing module.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "exp62", PROJECT / "phase four" / "exp62_harder_tasks.py")
    e62_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(e62_mod)
    all_problems = e62_mod.ALL_PROBLEMS
    print(f"  Loaded {len(all_problems)} problems from Exp 62 source\n")

    by_family = {}
    for p in all_problems:
        by_family.setdefault(p["family"], []).append(p)

    fams = list(by_family.keys())
    client = cheap("gemini")
    results_per_holdout = {}

    for held_out_family in fams:
        print(f"\n=== Hold out: {held_out_family} ===")
        train_families = [f for f in fams if f != held_out_family]
        train_examples_text = []
        for tf in train_families:
            opt = fam_optimal[tf]
            samples = by_family[tf][:2]  # 2 examples each
            for s in samples:
                train_examples_text.append(
                    f"  Problem: {s['prompt'][:200]}\n"
                    f"  Optimal prior: {opt} ({EXISTING_PRIORS[opt]})")
        train_text = "\n\n".join(train_examples_text)

        # Show 3 examples from held-out family without revealing optimal prior
        held_problems = by_family[held_out_family]
        new_problems_text = "\n\n".join(
            f"  Problem: {p['prompt'][:200]}" for p in held_problems[:3])

        # Generate new prior
        gen = generate_prior(client, train_text, new_problems_text,
                              list(EXISTING_PRIORS.keys()))
        print(f"  Generated prior: '{gen['name']}'")
        print(f"    text: {gen['prior_text'][:150]}")
        if not gen['prior_text']:
            print(f"  Failed to generate prior; skipping.")
            continue

        # Test on held-out family
        # Conditions: baseline, generated-new-prior, existing-prior-from-best-fixed
        # (in Exp 62, scheduler picks already in e62), oracle
        fam_problems = held_problems
        n = len(fam_problems)

        baseline_score = sum(pp[p["pid"]].get("baseline_score", 0) for p in fam_problems)
        oracle_prior = fam_optimal[held_out_family]
        oracle_score = sum(pp[p["pid"]].get(f"{oracle_prior}_score", 0) for p in fam_problems)

        # Solve with the new generated prior
        new_scores = []
        with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
            futs = {ex.submit(solve, client, p["prompt"], gen["prior_text"]): p
                      for p in fam_problems}
            for f in as_completed(futs):
                p = futs[f]; ans = f.result()
                ext = extract(ans)
                new_scores.append(sc(ext, p["gold"], p.get("tol")))
        new_acc = sum(new_scores) / n

        print(f"  Held-out family results (n={n}):")
        print(f"    baseline (no prior):           {baseline_score/n:.3f}")
        print(f"    new generated prior:           {new_acc:.3f}")
        print(f"    oracle (true optimal prior):   {oracle_score/n:.3f}")
        print(f"    delta vs baseline: {new_acc - baseline_score/n:+.3f}")
        print(f"    delta vs oracle:   {new_acc - oracle_score/n:+.3f}")

        results_per_holdout[held_out_family] = {
            "generated_prior_name": gen["name"],
            "generated_prior_text": gen["prior_text"],
            "n": n,
            "baseline_acc": baseline_score / n,
            "new_prior_acc": new_acc,
            "oracle_acc": oracle_score / n,
            "delta_vs_baseline": new_acc - baseline_score / n,
            "delta_vs_oracle": new_acc - oracle_score / n,
        }

    # Aggregate across hold-outs
    print(f"\n\n=== Aggregate Stage 4 evaluation ===")
    deltas_baseline = [r["delta_vs_baseline"] for r in results_per_holdout.values()]
    deltas_oracle = [r["delta_vs_oracle"] for r in results_per_holdout.values()]
    if deltas_baseline:
        avg_db = sum(deltas_baseline) / len(deltas_baseline)
        avg_do = sum(deltas_oracle) / len(deltas_oracle)
        print(f"  Mean delta vs baseline: {avg_db:+.3f} (positive = generated prior helps)")
        print(f"  Mean delta vs oracle:   {avg_do:+.3f} (closer to 0 = matches optimal)")
        n_positive = sum(1 for d in deltas_baseline if d > 0)
        n_total = len(deltas_baseline)
        print(f"  Generated prior beats baseline on {n_positive}/{n_total} hold-outs")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_holdouts": len(results_per_holdout),
           "results_per_holdout": results_per_holdout,
           "aggregate": {
               "mean_delta_vs_baseline": (sum(deltas_baseline) / len(deltas_baseline)
                                            if deltas_baseline else 0),
               "mean_delta_vs_oracle": (sum(deltas_oracle) / len(deltas_oracle)
                                          if deltas_oracle else 0),
               "n_positive_baseline": sum(1 for d in deltas_baseline if d > 0),
           } if deltas_baseline else {}}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
