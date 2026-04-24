"""Exp 37 — Cohen's κ cross-tier analysis.

Consumes per-pid verdicts from Exp 35 (expensive: opus, gpt5) and
Exp 36 (cheap: gemini, haiku, gpt-mini) on the same 4 candidates
× 50 pids, and computes pairwise Cohen's κ between all 5 judges.

Three partitions:
  within-cheap  = κ(gemini, haiku), κ(gemini, gpt-mini),
                   κ(haiku, gpt-mini)           (3 pairs)
  within-exp    = κ(opus, gpt5)                  (1 pair)
  cross-tier    = κ(cheap_i, expensive_j)        (6 pairs)

If cross-tier κ << within-tier κ, the judge-tier choice matters and
the audit stack's cheap-tier judges might have systematic tier-
specific biases.

Per candidate: report 10 κ values. Aggregate across 4 candidates:
mean κ per partition + 95% bootstrap CI.
"""

import json
import sys
from pathlib import Path
from itertools import combinations

PROJECT = Path(__file__).parent.parent
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp37_kappa_cross_tier_log.json"

# 5 judges
CHEAP = ["gemini", "claude_haiku", "gpt_mini"]
EXPENSIVE = ["claude_opus", "gpt5"]
ALL = CHEAP + EXPENSIVE

CATEGORIES = ["ext", "base", "tie"]


def kappa(v1, v2):
    """Cohen's κ between two dicts of {pid: verdict}. Returns (κ, n_common)."""
    common = [pid for pid in v1 if pid in v2
              and v1[pid] in CATEGORIES and v2[pid] in CATEGORIES]
    if not common:
        return None, 0
    n = len(common)
    # Observed agreement
    obs = sum(1 for pid in common if v1[pid] == v2[pid]) / n
    # Expected agreement (marginal product)
    exp = 0.0
    for c in CATEGORIES:
        p1 = sum(1 for pid in common if v1[pid] == c) / n
        p2 = sum(1 for pid in common if v2[pid] == c) / n
        exp += p1 * p2
    if exp >= 1.0:
        return 1.0, n
    return (obs - exp) / (1 - exp), n


def load_verdicts():
    """Return {cand_id: {family: {pid: verdict}}} merging exp35 + exp36."""
    out = {}
    exp35 = json.loads((AUTO_DIR / "exp35_expensive_extended_log.json").read_text())
    for r in exp35.get("results", []):
        cid = r["cand_id"]
        out.setdefault(cid, {})
        for fam, v in r["verdicts"].items():
            out[cid][fam] = v

    exp36 = json.loads((AUTO_DIR / "exp36_cheap_verdicts_log.json").read_text())
    for r in exp36.get("results", []):
        cid = r["cand_id"]
        out.setdefault(cid, {})
        for fam, v in r["verdicts"].items():
            out[cid][fam] = v
    return out


def main():
    data = load_verdicts()
    print(f"Candidates with data: {list(data)}")
    print(f"Judges per candidate: {[len(data[c]) for c in data]}")

    # Per-candidate pairwise κ
    partition = {"within_cheap": [], "within_exp": [], "cross_tier": []}
    per_cand = {}
    for cid, fams in data.items():
        present = [f for f in ALL if f in fams]
        if len(present) < 2:
            continue
        per_cand[cid] = {}
        for f1, f2 in combinations(present, 2):
            k, n = kappa(fams[f1], fams[f2])
            if k is None:
                continue
            per_cand[cid][(f1, f2)] = (k, n)
            if f1 in CHEAP and f2 in CHEAP:
                partition["within_cheap"].append(k)
            elif f1 in EXPENSIVE and f2 in EXPENSIVE:
                partition["within_exp"].append(k)
            else:
                partition["cross_tier"].append(k)

    print(f"\n=== Per-candidate pairwise κ ===")
    for cid, pairs in per_cand.items():
        print(f"\n  {cid}:")
        for (f1, f2), (k, n) in sorted(pairs.items()):
            tier = "cheap-cheap" if f1 in CHEAP and f2 in CHEAP \
                else "exp-exp" if f1 in EXPENSIVE and f2 in EXPENSIVE \
                else "cross"
            print(f"    {f1:12s} × {f2:12s} [{tier:10s}]  κ = {k:+.3f}  (n={n})")

    print(f"\n=== Aggregate κ by partition ===")
    for part, vals in partition.items():
        if not vals:
            print(f"  {part:15s} (no data)")
            continue
        mean_k = sum(vals) / len(vals)
        # crude min/max
        print(f"  {part:15s} mean κ = {mean_k:+.3f}  "
              f"[{min(vals):+.3f}, {max(vals):+.3f}]  "
              f"(n = {len(vals)} pairs across candidates)")

    # Interpretation
    wc = partition["within_cheap"]
    we = partition["within_exp"]
    ct = partition["cross_tier"]
    if wc and ct:
        delta = (sum(wc)/len(wc)) - (sum(ct)/len(ct))
        print(f"\n  Δ(within-cheap − cross-tier) = {delta:+.3f}")
        if abs(delta) < 0.1:
            verdict = "cross-tier κ comparable to within-tier → stability assumption holds"
        elif delta > 0.15:
            verdict = "cross-tier κ meaningfully lower → stability assumption violated"
        else:
            verdict = "cross-tier κ slightly lower → weak evidence against stability"
        print(f"  Verdict: {verdict}")

    out = {
        "per_candidate": {
            cid: {f"{f1}__{f2}": {"kappa": k, "n": n}
                   for (f1, f2), (k, n) in pairs.items()}
            for cid, pairs in per_cand.items()
        },
        "partition_means": {
            p: {"mean_kappa": sum(vs) / len(vs) if vs else None,
                 "n_pairs": len(vs),
                 "values": vs}
            for p, vs in partition.items()
        },
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
