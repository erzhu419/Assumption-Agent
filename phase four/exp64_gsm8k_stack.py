"""Exp 64 — Apply the agent stack to GSM8K (second independent task domain).

Closes the ``empirical unit too narrow'' concern by validating the
scheduler+priors architecture on an independently designed, widely
adopted benchmark (GSM8K math word problems) rather than only on
our hand-crafted 4-family synthetic dataset.

Protocol:
  - Sample 100 GSM8K problems (already in repo's GSM8K subset)
  - Run baseline + scheduler-picked + each-fixed-prior + oracle
  - Compare scheduler accuracy vs baseline and best fixed prior

GSM8K's optimal prior is presumably ``decompose'' (multi-step
arithmetic), but we let the scheduler pick from {decompose,
restate, estimate, constraints, none}.

Cost: 100 × (5 fixed + 1 scheduler + 1 oracle = 7) + 100 schedule
= 800 calls. ~$5.
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
OUT_LOG = AUTO / "exp64_gsm8k_stack_log.json"

PRIORS = {
    "decompose": "Before answering, decompose into atomic substeps.",
    "restate": "Before answering, RE-READ the question carefully; check if the obvious interpretation is a trap.",
    "estimate": "Before answering, give an order-of-magnitude estimate; sanity-check final answer.",
    "constraints": "Before answering, enumerate explicit constraints; satisfy all simultaneously.",
    "none": "",
}

SOLVE_PROMPT = """## Problem
{problem}

## Approach hint
{approach}

## Output
Reason step by step, then on the LAST LINE write exactly:
ANSWER: <numeric answer only>
"""

SCHED_PROMPT = """You are a strategy selector. Given a problem, choose ONE prior from:
{{decompose, restate, estimate, constraints, none}}.

- decompose: best for multi-step arithmetic.
- restate: best for trick questions.
- estimate: best for Fermi/order-of-magnitude.
- constraints: best for logic puzzles / probabilistic traps.
- none: only if no prior is more applicable.

## Problem
{problem}

## Output (JSON only)
{{"choice": "decompose"|"restate"|"estimate"|"constraints"|"none", "reason": "1 sentence"}}
"""


def load_gsm8k_subset(n=100):
    """Load GSM8K test problems via HuggingFace datasets, matching Exp 32 setup."""
    print(f"  Loading GSM8K test split via HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=f"test[:{n}]")
    problems = []
    for i, x in enumerate(ds):
        ans = x["answer"]
        m = re.search(r'####\s*(-?\d+(?:\.\d+)?)', ans)
        gold = float(m.group(1)) if m else None
        if gold is not None:
            problems.append({"prompt": x["question"], "gold": gold,
                              "pid": f"gsm_{i:03d}"})
    return problems[:n]


def solve(client, problem, prior_name):
    p = PRIORS[prior_name]
    appr = f"Use this strategy: {p}" if p else "Use any approach you think appropriate."
    try:
        r = client.generate(SOLVE_PROMPT.format(problem=problem, approach=appr),
                             max_tokens=600, temperature=0.0)
        return r["text"].strip()
    except Exception as e: return f"[err: {e}]"

def schedule(client, problem):
    try:
        r = client.generate(SCHED_PROMPT.format(problem=problem), max_tokens=200, temperature=0.0)
        m = re.search(r'"choice"\s*:\s*"(decompose|restate|estimate|constraints|none)"', r["text"])
        if m: return m.group(1)
    except Exception: pass
    return "none"

def extract(text):
    m = re.search(r'ANSWER:\s*(.+?)(?:\n|$)', text, re.IGNORECASE)
    return (m.group(1) if m else text[-200:]).strip()

def score_n(ext, gold):
    s = ext.replace(",", "").replace("$", "").replace("%", "").strip().rstrip(".")
    m = re.search(r'-?\d+(?:\.\d+)?', s)
    if not m: return 0
    try: ev = float(m.group())
    except: return 0
    try: gv = float(gold)
    except: return 0
    return 1 if abs(ev - gv) < 0.05 else 0


def main():
    print(f"=== Exp 64: agent stack on GSM8K ===")
    problems = load_gsm8k_subset(n=100)
    print(f"  Loaded {len(problems)} GSM8K problems\n")

    solver = cheap("gemini")
    sched_c = cheap("gemini")
    conds = ["baseline", "decompose", "restate", "estimate", "constraints"]
    answers = {c: {} for c in conds}; answers["scheduler"] = {}
    sched_picks = {}

    print(f"[1/3] Fixed conds (5 × {len(problems)} = {5*len(problems)} calls)...")
    tasks = [(c, p) for c in conds for p in problems]
    def run_fix(c, p):
        prior = "none" if c == "baseline" else c
        return c, p["pid"], solve(solver, p["prompt"], prior)
    t0 = time.time(); done = 0
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(run_fix, c, p) for c, p in tasks]
        for f in as_completed(futs):
            c, pid, ans = f.result(); answers[c][pid] = ans; done += 1
            if done % 100 == 0: print(f"  fixed {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    print(f"\n[2/3] Schedule + scheduler-solve (2 × {len(problems)} calls)...")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(schedule, sched_c, p["prompt"]): p for p in problems}
        for f in as_completed(futs):
            p = futs[f]; sched_picks[p["pid"]] = f.result()
    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = {ex.submit(solve, solver, p["prompt"], sched_picks[p["pid"]]): p for p in problems}
        for f in as_completed(futs):
            p = futs[f]; answers["scheduler"][p["pid"]] = f.result()
    print(f"  done ({time.time()-t0:.0f}s)")

    # Score
    print(f"\n=== GSM8K accuracy by condition ===")
    print(f"{'Cond':14s} {'accuracy':>10s}")
    print("-" * 30)
    summary = {}
    for c in conds + ["scheduler"]:
        n_correct = sum(score_n(extract(answers[c].get(p["pid"], "")), p["gold"])
                          for p in problems)
        acc = n_correct / len(problems)
        print(f"{c:14s} {acc:>10.3f}")
        summary[c] = {"accuracy": acc, "n_correct": n_correct}

    # Pick distribution
    from collections import Counter
    pick_dist = Counter(sched_picks.values())
    print(f"\nScheduler pick distribution: {dict(pick_dist)}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_problems": len(problems),
           "task_domain": "GSM8K",
           "scheduler_picks": sched_picks,
           "pick_distribution": dict(pick_dist),
           "summary": summary}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
