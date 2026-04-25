"""Exp 57 — Stage 0.5 cheap world model: predict-then-execute.

The reviewer's standing concern about §Roadmap is that ``no Stage 0.5
world model is implemented or empirically tested.'' This experiment
implements the simplest possible Stage 0.5 and calibrates it against
Exp 56's ground truth.

The world model is a CHEAP, SEPARATE LLM call that, given
(problem, prior), predicts the probability that applying the prior
will produce a correct answer --- without actually solving. The
intended use is to screen many candidate priors before paying the
full cost of execution. To be useful, the predictor must be
non-trivially calibrated against actual outcomes.

We use Exp 56's data: for each of 60 problems and each of 4 priors
(decompose, restate, estimate, constraints), we have the actual
correctness from running the solver. The world model receives the
problem + the prior text, predicts P(success) in [0, 1], and we
compute calibration metrics:

  Brier score:  mean (predicted - actual)^2  (lower = better)
  AUROC:        rank-based discrimination (>0.5 = better than chance)
  ECE:          expected calibration error in 5 bins
  Top-1 match:  if scheduler picks argmax_prior(predicted), how
                often does that pick match the actual best prior?

Cost: 4 priors × 60 problems = 240 predictor calls. ~$2.
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
OUT_LOG = AUTO / "exp57_stage05_worldmodel_log.json"

PRIORS = {
    "decompose": "decompose the problem into atomic substeps",
    "restate":   "re-read the question carefully; check for traps in the obvious interpretation",
    "estimate":  "produce an order-of-magnitude estimate first; sanity-check final answer",
    "constraints": "enumerate explicit constraints; satisfy all simultaneously",
}

PREDICT_PROMPT = """You are a hypothesis-evaluator. Given a problem and a candidate
methodological strategy, predict the probability (a number in [0, 1]) that
applying this strategy will produce the correct answer when executed by a
small LLM.

You should think about: does the strategy match the problem type? Will it
help or distract? Is the strategy a known trap for this kind of problem?

Do NOT solve the problem. Only estimate the probability.

## Problem
{problem}

## Candidate strategy
{prior_text}

## Output (JSON only)
{{"p_success": <float in [0,1]>, "reason": "1 short sentence"}}
"""

def predict(client, problem, prior_text):
    try:
        r = client.generate(PREDICT_PROMPT.format(problem=problem, prior_text=prior_text),
                             max_tokens=200, temperature=0.0)
        m = re.search(r'"p_success"\s*:\s*(\d+(?:\.\d+)?)', r["text"])
        if m:
            v = float(m.group(1))
            return max(0.0, min(1.0, v))
    except Exception: pass
    return 0.5

def brier(preds, actuals):
    return sum((p - a) ** 2 for p, a in zip(preds, actuals)) / max(len(preds), 1)

def auroc(preds, actuals):
    pairs = sorted(zip(preds, actuals), key=lambda x: x[0])
    n_pos = sum(actuals); n_neg = len(actuals) - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    rank_sum_pos = 0
    for i, (p, a) in enumerate(pairs):
        if a == 1: rank_sum_pos += i + 1
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def ece(preds, actuals, n_bins=5):
    bins = [(i / n_bins, (i + 1) / n_bins) for i in range(n_bins)]
    total = 0
    for lo, hi in bins:
        mask = [(lo <= p < hi) or (hi == 1.0 and p == 1.0) for p in preds]
        if not any(mask): continue
        bp = [p for p, m in zip(preds, mask) if m]
        ba = [a for a, m in zip(actuals, mask) if m]
        avg_p = sum(bp) / len(bp); avg_a = sum(ba) / len(ba)
        total += (len(bp) / len(preds)) * abs(avg_p - avg_a)
    return total

def main():
    print(f"=== Exp 57: Stage 0.5 world model calibration on Exp 56 ground truth ===")
    e56 = json.loads((AUTO / "exp56_stage1_broader_log.json").read_text())
    print(f"  Loaded Exp 56: {e56['n_problems']} problems")

    pp = e56["per_problem"]
    problems_data = []
    for pid, d in pp.items():
        problems_data.append({"pid": pid, "prompt": d["prompt"]})
    print(f"  {len(problems_data)} problems to predict on, 4 priors each = "
          f"{len(problems_data)*4} calls\n")

    client = cheap("gemini")
    print(f"[1/2] Predicting P(success) per (problem, prior)...")
    preds = {}
    tasks = [(p, prior) for p in problems_data for prior in PRIORS]
    t0 = time.time(); done = 0

    def run(p, prior):
        return p["pid"], prior, predict(client, p["prompt"], PRIORS[prior])

    with ThreadPoolExecutor(max_workers=PARALLEL) as ex:
        futs = [ex.submit(run, p, pr) for p, pr in tasks]
        for f in as_completed(futs):
            pid, prior, p_pred = f.result()
            preds[(pid, prior)] = p_pred
            done += 1
            if done % 30 == 0: print(f"  predict {done}/{len(tasks)} ({time.time()-t0:.0f}s)")

    print(f"\n[2/2] Calibration metrics...")
    flat_preds, flat_actuals = [], []
    per_prior_preds = {prior: [] for prior in PRIORS}
    per_prior_actuals = {prior: [] for prior in PRIORS}
    per_problem_picks = {}
    for pid, d in pp.items():
        scores_by_prior = {}
        preds_by_prior = {}
        for prior in PRIORS:
            actual = d.get(f"{prior}_score", 0)
            pred = preds.get((pid, prior), 0.5)
            flat_preds.append(pred); flat_actuals.append(actual)
            per_prior_preds[prior].append(pred); per_prior_actuals[prior].append(actual)
            scores_by_prior[prior] = actual; preds_by_prior[prior] = pred
        # Pick the prior with highest predicted success
        wm_pick = max(preds_by_prior.items(), key=lambda x: x[1])[0]
        # Best-actual-prior (oracle)
        best_actual_prior = max(scores_by_prior.items(), key=lambda x: x[1])[0]
        per_problem_picks[pid] = {"world_model_pick": wm_pick,
                                    "best_actual_prior": best_actual_prior,
                                    "preds": preds_by_prior,
                                    "actuals": scores_by_prior,
                                    "family": d["family"]}

    overall_brier = brier(flat_preds, flat_actuals)
    overall_auroc = auroc(flat_preds, flat_actuals)
    overall_ece = ece(flat_preds, flat_actuals)

    print(f"\n=== Calibration of cheap world model on Exp 56 ground truth ===")
    print(f"  Brier score: {overall_brier:.4f}  (uniform 0.5 baseline = 0.25; "
          f"lower = better)")
    print(f"  AUROC:       {overall_auroc:.4f}  (0.5 = chance; >0.7 = useful)")
    print(f"  ECE:         {overall_ece:.4f}  (lower = better calibrated)")
    print(f"\n  Per-prior Brier scores:")
    for prior in PRIORS:
        b = brier(per_prior_preds[prior], per_prior_actuals[prior])
        a = auroc(per_prior_preds[prior], per_prior_actuals[prior])
        print(f"    {prior:14s} Brier={b:.4f}  AUROC={a:.4f}")

    # Top-1 prior selection: how often does world-model argmax match
    # the actual-best prior?
    print(f"\n=== Top-1 prior-selection accuracy ===")
    correct = 0
    family_correct = {f: 0 for f in ["A", "B", "C", "D"]}
    family_count = {f: 0 for f in ["A", "B", "C", "D"]}
    for pid, info in per_problem_picks.items():
        f = info["family"][0]
        family_count[f] += 1
        if info["world_model_pick"] == info["best_actual_prior"]:
            correct += 1
            family_correct[f] += 1
    print(f"  World model's argmax matches best-actual-prior: "
          f"{correct}/60 = {correct/60:.1%}")
    print(f"  Random baseline: 1/4 = 25%")
    for f in ["A", "B", "C", "D"]:
        if family_count[f] > 0:
            print(f"  Family {f}: {family_correct[f]}/{family_count[f]} = "
                  f"{family_correct[f]/family_count[f]:.1%}")

    # If the world model picks the OPTIMAL prior for each problem,
    # what's the achieved accuracy?
    fam_optimal = {"A": "decompose", "B": "restate", "C": "estimate",
                    "D": "constraints"}
    accuracy_under_wm_pick = sum(per_problem_picks[pid]["actuals"][per_problem_picks[pid]["world_model_pick"]]
                                    for pid in per_problem_picks) / 60
    accuracy_under_oracle_pick = sum(per_problem_picks[pid]["actuals"][fam_optimal[per_problem_picks[pid]["family"][0]]]
                                        for pid in per_problem_picks) / 60
    print(f"\n  If we use world-model argmax as the scheduler: "
          f"accuracy = {accuracy_under_wm_pick:.3f}")
    print(f"  Family-oracle (true optimal prior per family): "
          f"accuracy = {accuracy_under_oracle_pick:.3f}")

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "n_predictions": len(flat_preds),
           "metrics": {
               "brier": overall_brier, "auroc": overall_auroc, "ece": overall_ece,
               "per_prior_brier": {p: brier(per_prior_preds[p], per_prior_actuals[p])
                                     for p in PRIORS},
               "per_prior_auroc": {p: auroc(per_prior_preds[p], per_prior_actuals[p])
                                     for p in PRIORS},
               "top1_pick_accuracy": correct / 60,
               "accuracy_under_wm_pick": accuracy_under_wm_pick,
               "accuracy_under_oracle_pick": accuracy_under_oracle_pick,
           },
           "per_problem_picks": per_problem_picks}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
