"""Exp 82 v2 step C: redesigned `feature` kind evaluator.

The original ablation evaluator treated `feature` as a no-op for correctness
(ext_answer = base_answer), so every feature hypothesis got Δ_ext_base = 0
and was rejected. That's wrong: a `feature` is a TRIGGER ROUTER, not a
solver-modifier. Its job is to detect "should this wisdom apply?", and the
right metric is **selectivity** — high fire-rate on SHOULD_FIRE problems,
low fire-rate on NO_FIRE problems.

Two diagnostics:

  1. Selectivity (no LLM calls, runs on cached forensic / hypotheses.jsonl)
       fire_rate_trigger     = (#fire in SHOULD_FIRE pids) / |SHOULD_FIRE|
       fire_rate_outside     = (#fire in NO_FIRE   pids) / |NO_FIRE|
       fire_rate_neutral     = (#fire in NEUTRAL   pids) / |NEUTRAL|
       selectivity_lift      = fire_rate_trigger - fire_rate_outside
       selectivity_ratio     = fire_rate_trigger / max(fire_rate_outside, 0.01)
       precision (vs label)  = TP/(TP+FP)  where TP=fire∩SHOULD_FIRE
       recall    (vs label)  = TP/(TP+FN)
       f1                    = 2·P·R/(P+R)

     Accept criterion v2 for `feature`:
       selectivity_lift ≥ 0.30  AND
       fire_rate_outside ≤ 0.30  AND
       fire_rate_trigger ≥ 0.50

  2. Composition: feature × {constraint, decomposition, verification, hp_change}
     For a feature F and structural-kind hypothesis K from the same wisdom W:
       gated_answer(pid) = K_ext_answer if F fires on pid else BASE_answer
     Compare correctness of gated_answer vs uniformly-applied K.
     If gated > uniform on trigger AND gated ≥ uniform on outside, the feature
     adds value as a router. (This requires re-grading, with LLM — runs only
     if --with-composition is passed.)

Usage:
  python3 feature_eval_v2.py                 # selectivity only, all *.jsonl files
  python3 feature_eval_v2.py --hypos FILE    # use specific hypotheses.jsonl
  python3 feature_eval_v2.py --with-composition  # also test feature×kind gating

Output: feature_eval_v2_summary.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT / "phase zero" / "scripts"))

EXP_DIR = Path(__file__).parent
GOLD = EXP_DIR / "gold_answers.json"
TRIGGER_LABELS = PROJECT / "phase four" / "autonomous" / "exp17_trigger_labels.json"


def detect_feature(problem_text: str, expr: dict) -> bool:
    """Apply the feature's keyword/regex detector to a problem text."""
    kws_zh = expr.get("keywords_zh", []) or []
    kws_en = expr.get("keywords_en", []) or []
    regs = expr.get("regex", []) or []
    for kw in kws_zh + kws_en:
        if kw and kw in problem_text:
            return True
    for r in regs:
        if r:
            try:
                if re.search(r, problem_text):
                    return True
            except re.error:
                continue
    return False


def selectivity_metrics(fired_by_label: dict) -> dict:
    """fired_by_label: {label: (n_fire, n_total)} for SHOULD_FIRE/NO_FIRE/NEUTRAL.

    Returns selectivity diagnostics including precision/recall/f1.
    """
    sf = fired_by_label.get("SHOULD_FIRE", (0, 0))
    nf = fired_by_label.get("NO_FIRE", (0, 0))
    nu = fired_by_label.get("NEUTRAL", (0, 0))

    fr_trig = (sf[0] / sf[1]) if sf[1] else 0.0
    fr_out = (nf[0] / nf[1]) if nf[1] else 0.0
    fr_neu = (nu[0] / nu[1]) if nu[1] else 0.0

    tp = sf[0]
    fp = nf[0] + nu[0]
    fn = sf[1] - sf[0]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "fire_rate_should_fire": fr_trig,
        "fire_rate_no_fire": fr_out,
        "fire_rate_neutral": fr_neu,
        "selectivity_lift": fr_trig - fr_out,
        "selectivity_ratio": (fr_trig / fr_out) if fr_out > 0 else float("inf") if fr_trig > 0 else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_should_fire": sf[1], "n_no_fire": nf[1], "n_neutral": nu[1],
        "tp_should_fire_fired": sf[0],
        "fp_no_fire_fired": nf[0],
    }


def feature_decision(metrics: dict) -> tuple[str, str]:
    """Apply v2 accept criterion."""
    if metrics["selectivity_lift"] >= 0.30 and metrics["fire_rate_no_fire"] <= 0.30 and metrics["fire_rate_should_fire"] >= 0.50:
        return "accepted", None
    reasons = []
    if metrics["selectivity_lift"] < 0.30:
        reasons.append(f"low_lift({metrics['selectivity_lift']:+.2f})")
    if metrics["fire_rate_no_fire"] > 0.30:
        reasons.append(f"too_promiscuous(out={metrics['fire_rate_no_fire']:.0%})")
    if metrics["fire_rate_should_fire"] < 0.50:
        reasons.append(f"low_recall(trig={metrics['fire_rate_should_fire']:.0%})")
    return "rejected", "+".join(reasons) or "uncategorized"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hypos", default=None,
                    help="Path to hypotheses.jsonl (defaults to hypotheses.jsonl, or hypotheses_small3.jsonl if main is missing)")
    ap.add_argument("--out", default="feature_eval_v2_summary.json")
    args = ap.parse_args()

    if args.hypos:
        hypo_path = Path(args.hypos)
        if not hypo_path.is_absolute():
            hypo_path = (EXP_DIR / hypo_path).resolve()
    else:
        candidates = ["hypotheses.jsonl", "hypotheses_small3.jsonl"]
        for c in candidates:
            p = EXP_DIR / c
            if p.exists() and p.stat().st_size > 0:
                hypo_path = p
                break
        else:
            raise RuntimeError("no hypotheses file found")

    print(f"Loading hypotheses from {hypo_path.relative_to(PROJECT)}", flush=True)
    hypos = [json.loads(l) for l in open(hypo_path) if l.strip()]
    print(f"  total: {len(hypos)} hypotheses", flush=True)

    feature_hypos = [h for h in hypos if h["kind"] == "feature"]
    print(f"  feature kind: {len(feature_hypos)}", flush=True)

    print(f"\nLoading gold answers ({GOLD.relative_to(PROJECT)}) and trigger labels...", flush=True)
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    labels = json.loads(TRIGGER_LABELS.read_text(encoding="utf-8"))

    # ── Per feature hypothesis: run detector on ALL 50 problems and ────
    # compute selectivity vs exp17 labels.
    summary = {"hypotheses": [], "by_seed": {}}
    print(f"\n=== Feature kind v2 selectivity ===\n", flush=True)
    print(f"  {'wisdom/feature':32s} | {'fire SHOULD':>12s} | {'fire NO':>9s} | {'fire NEUT':>10s} | {'lift':>6s} | {'P':>4s} | {'R':>4s} | {'F1':>4s} | {'decision':>10s}", flush=True)
    print(f"  {'-'*32}-+-{'-'*12}-+-{'-'*9}-+-{'-'*10}-+-{'-'*6}-+-{'-'*4}-+-{'-'*4}-+-{'-'*4}-+-{'-'*10}", flush=True)
    for h in feature_hypos:
        cid = h["seed_cid"]
        cid_labels = labels.get(cid, {})
        # bucket by label
        bucket: dict = defaultdict(lambda: [0, 0])
        for pid, rec in cid_labels.items():
            verdict = rec.get("verdict", "NA")
            if pid not in gold:
                continue
            problem = gold[pid]["description"]
            fired = detect_feature(problem, h["expr"])
            if fired:
                bucket[verdict][0] += 1
            bucket[verdict][1] += 1
        m = selectivity_metrics({k: tuple(v) for k, v in bucket.items()})
        decision, reason = feature_decision(m)
        out = {
            "hid": h["hid"], "seed_cid": cid, "claim": h["claim"][:120],
            "expr_keywords_zh": h["expr"].get("keywords_zh", []),
            "expr_n_regex": len(h["expr"].get("regex", [])),
            **m,
            "decision": decision, "reason": reason,
        }
        summary["hypotheses"].append(out)
        print(f"  {cid+'/feature':32s} | {m['tp_should_fire_fired']:2d}/{m['n_should_fire']:2d} ({m['fire_rate_should_fire']:>4.0%}) | {m['fp_no_fire_fired']:2d}/{m['n_no_fire']:2d} ({m['fire_rate_no_fire']:>3.0%}) | {bucket['NEUTRAL'][0]:2d}/{m['n_neutral']:2d} ({m['fire_rate_neutral']:>3.0%}) | {m['selectivity_lift']:+5.2f} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {decision:>10s}{' ('+reason+')' if reason else ''}", flush=True)

    # Aggregate
    n_acc = sum(1 for r in summary["hypotheses"] if r["decision"] == "accepted")
    print(f"\n  → feature kind accept rate: {n_acc}/{len(summary['hypotheses'])} ({n_acc/max(1,len(summary['hypotheses'])):.0%})", flush=True)
    print(f"  → mean selectivity lift: {sum(r['selectivity_lift'] for r in summary['hypotheses'])/max(1,len(summary['hypotheses'])):+.2f}", flush=True)
    print(f"  → mean F1: {sum(r['f1'] for r in summary['hypotheses'])/max(1,len(summary['hypotheses'])):.2f}", flush=True)

    out_path = EXP_DIR / args.out
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nSaved → {out_path.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
