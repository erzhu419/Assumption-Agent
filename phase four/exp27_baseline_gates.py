"""Exp 27 — Baseline gates for discrimination control.

Addresses reviewer objection #11:
  "No computational baseline; is your complex v2 gate better than
   accepting everything? we don't know."

Fix: compare researcher's Exp 17 gate + agent's Exp 24 gate against:
  - Always-accept (12/12)
  - Always-reject (0/12)
  - Random 33% (expected 4/12, CI via resampling)
  - Random 25% (expected 3/12)
  - "Survives Exp 11 triangulation" (majority-of-3 families at n=50)

The triangulation baseline (Exp 11) is our closest-to-ground-truth:
  Exp 11 showed W078 and WCAND07 pass majority-of-3 at n=50.
  Treat that as pseudo-ground-truth for precision/recall.
"""

import json
import random
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).parent.parent
AUTO_DIR = PROJECT / "phase four" / "autonomous"


def load_exp_pass_sets():
    e17 = json.loads((AUTO_DIR / "exp17_trigger_conditioned_log.json").read_text())[-1]["results"]
    exp17_pass = {r["cid"] for r in e17 if r.get("gate_pass")}

    exp24 = json.loads((AUTO_DIR / "exp24_agent_v2_gate_log.json").read_text())[-1]
    exp24_pass = set(exp24["agent_v2_pass"])

    # Exp 11 triangulation pseudo-ground-truth: candidates majority-of-3 at n=50
    # From paper table §4.10: W078 and WCAND07 passed 2/3 threshold at 0.55
    # (W076: 1/3, W077: 1/3, WCAND07: 2/3, WCAND11: 1/3, W078: 2/3)
    triangulation_pass = {"WCROSSL01", "WCAND07"}

    return {
        "exp17_researcher": exp17_pass,
        "exp24_agent_v2":   exp24_pass,
        "triangulation_gt": triangulation_pass,
    }


def all_cids():
    return [f"WCAND{i:02d}" for i in range(1, 12)] + ["WCROSSL01"]


def precision_recall_f1(pred_set, gt_set, all_items):
    pred = set(pred_set) & set(all_items)
    gt = set(gt_set) & set(all_items)
    tp = len(pred & gt)
    fp = len(pred - gt)
    fn = len(gt - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*p*r / (p+r) if (p+r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn,
            "n_pred": len(pred), "n_gt": len(gt)}


def random_baseline(pass_fraction, all_items, gt_set, n_iters=10000):
    """Monte-carlo: on each iter, randomly select PASS candidates per rate."""
    rng = random.Random(42)
    n_pass = round(len(all_items) * pass_fraction)
    f1s = []; ps = []; rs = []
    for _ in range(n_iters):
        pred = set(rng.sample(all_items, n_pass))
        m = precision_recall_f1(pred, gt_set, all_items)
        f1s.append(m["f1"]); ps.append(m["precision"]); rs.append(m["recall"])
    return {"expected_pass": n_pass,
            "mean_precision": sum(ps)/len(ps),
            "mean_recall": sum(rs)/len(rs),
            "mean_f1": sum(f1s)/len(f1s),
            "p95_f1": sorted(f1s)[int(0.95*len(f1s))],
            "n_iters": n_iters}


def main():
    sets = load_exp_pass_sets()
    cids = all_cids()
    gt = sets["triangulation_gt"]  # pseudo-ground-truth
    print(f"Pseudo-ground-truth (Exp 11 majority-of-3 at n=50): {sorted(gt)}")
    print(f"All candidates: {cids}\n")

    rows = []

    # always-accept
    always = set(cids)
    m = precision_recall_f1(always, gt, cids)
    rows.append(("always_accept", m))

    # always-reject
    m = precision_recall_f1(set(), gt, cids)
    rows.append(("always_reject", m))

    # random baselines
    for frac, label in [(0.25, "random_25%"), (0.33, "random_33%"),
                        (0.50, "random_50%"), (0.67, "random_67%")]:
        r = random_baseline(frac, cids, gt)
        rows.append((label, {"precision": r["mean_precision"],
                              "recall": r["mean_recall"],
                              "f1": r["mean_f1"],
                              "n_pred": r["expected_pass"],
                              "n_gt": len(gt), "p95_f1": r["p95_f1"]}))

    # researcher + agent
    for name, key in [("researcher_exp17", "exp17_researcher"),
                      ("agent_v2_exp24", "exp24_agent_v2")]:
        m = precision_recall_f1(sets[key], gt, cids)
        rows.append((name, m))

    # Print
    print(f"{'gate':20s} {'precision':10s} {'recall':8s} {'f1':6s} {'n_pred':7s} {'extra'}")
    print("-" * 80)
    for name, m in rows:
        extra = ""
        if "p95_f1" in m: extra = f"p95_f1={m['p95_f1']:.2f}"
        print(f"  {name:18s} {m['precision']:<10.2f} {m['recall']:<8.2f} "
              f"{m['f1']:<6.2f} {m['n_pred']:<7d} {extra}")

    out = {
        "ground_truth": sorted(gt),
        "all_candidates": cids,
        "gates": {name: m for name, m in rows},
    }
    (AUTO_DIR / "exp27_baseline_gates_log.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → exp27_baseline_gates_log.json")


if __name__ == "__main__":
    main()
