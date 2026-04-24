"""Exp 28 — Rerun agent v2 gate with Exp 25 majority-vote labels.

Also drops the S_anti component (κ=0.14 in Exp 25 means the subset
labels are unreliable noise).  Agent's combination rule gets
downgraded accordingly.

Compares 3 gates on the same 12 candidates:
  1. Exp 24 (single-family Claude labels)
  2. Exp 28A (3-family majority-vote labels, keep all 4 components)
  3. Exp 28B (3-family majority-vote labels, drop S_anti component)

And compares all 3 against:
  - Exp 17 researcher gate
  - Triangulation pseudo-ground-truth (W078, WCAND07)
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT / "phase four"))

from exp21_data_api import list_candidates, candidate_info, per_pid_records

CACHE = PROJECT / "phase two" / "analysis" / "cache"
AUTO_DIR = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO_DIR / "exp28_majority_vote_gate_log.json"


# ---- shared text utilities (same as exp24) ----

def tokenize(text):
    if not text: return []
    out = []
    for chunk in re.findall(r"[A-Za-z0-9_一-鿿]+", text.lower()):
        if re.search(r"[一-鿿]", chunk):
            out.extend(list(chunk))
        else: out.append(chunk)
    return out

def char_ngrams(text, n=3):
    c = Counter()
    for i in range(max(0, len(text) - n + 1)): c[text[i:i+n]] += 1
    return c

def word_ngrams(tokens, n=1):
    c = Counter()
    for i in range(max(0, len(tokens) - n + 1)): c[" ".join(tokens[i:i+n])] += 1
    return c

def cosine(a, b):
    if not a or not b: return 0.0
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0) * b.get(k, 0) for k in keys)
    na = (sum(v*v for v in a.values())) ** 0.5
    nb = (sum(v*v for v in b.values())) ** 0.5
    if na < 1e-12 or nb < 1e-12: return 0.0
    return dot / (na * nb)

def text_sim(a, b):
    if not a or not b: return 0.0
    ta, tb = tokenize(a), tokenize(b)
    sc = cosine(char_ngrams(a), char_ngrams(b))
    su = cosine(word_ngrams(ta, 1), word_ngrams(tb, 1))
    sbi = cosine(word_ngrams(ta, 2), word_ngrams(tb, 2))
    return 0.3 * sc + 0.4 * su + 0.3 * sbi

def text_dist(a, b): return max(0.0, min(1.0, 1.0 - text_sim(a, b)))

def median(vals):
    if not vals: return 0.0
    s = sorted(vals); n = len(s)
    return s[n//2] if n % 2 == 1 else (s[n//2-1] + s[n//2]) / 2.0


# ---- conditioned evaluators ----

def eval_reframe(cid, labels):
    rows = per_pid_records(cid)
    dists = [text_dist(r["problem"], r["ext_what_changed"]) for r in rows
              if labels.get(r["pid"], {}).get("S_reframe")
              and r["problem"] and r["ext_what_changed"]]
    med = median(dists)
    return {"median_dist": med, "n": len(dists),
            "passed": med >= 0.30 and len(dists) >= 5}

def eval_delta(cid, labels):
    rows = per_pid_records(cid)
    dists = [text_dist(r["base_answer"], r["ext_answer"]) for r in rows
              if labels.get(r["pid"], {}).get("S_delta")
              and r["base_answer"] and r["ext_answer"]]
    med = median(dists)
    frac = sum(1 for d in dists if d >= 0.15) / len(dists) if dists else 0.0
    return {"median_dist": med, "fraction_above_0.15": frac, "n": len(dists),
            "passed": med >= 0.25 and frac >= 0.65 and len(dists) >= 5}

def eval_wisdom(cid, labels):
    info = candidate_info(cid)
    unpacked = info["unpacked"]
    if not unpacked: return {"passed": False, "n": 0}
    rows = per_pid_records(cid)
    sims = [text_sim(unpacked, r["problem"]) for r in rows
             if labels.get(r["pid"], {}).get("S_wisdom") and r["problem"]]
    if len(sims) < 3: return {"passed": False, "n": len(sims), "mean_sim": 0.0}
    mu = sum(sims) / len(sims)
    strong = sum(1 for s in sims if s > mu)
    return {"mean_sim": mu, "n": len(sims), "strong_count": strong,
            "passed": mu >= 0.08 and strong >= len(sims) * 0.3 and len(sims) >= 5}

def eval_anti(cid, labels):
    rows = per_pid_records(cid)
    rates = []
    for r in rows:
        if not labels.get(r["pid"], {}).get("S_anti"): continue
        aps = r.get("ext_anti_patterns", [])
        if not aps: continue
        ans = (r.get("ext_answer") or "").lower()
        hits = total = 0
        for ap in aps:
            if not isinstance(ap, str): continue
            total += 1
            if ap.strip()[:6].lower() in ans: hits += 1
        if total: rates.append(1.0 - hits/total)
    med = median(rates) if rates else 0.0
    return {"median_avoidance": med, "n": len(rates),
            "passed": med >= 0.65 and len(rates) >= 5}


# ---- combination rules ----

def combine_full(c):
    """Agent's original 4-component rule."""
    if not c["anti"]["passed"]: return False
    if not c["wisdom"]["passed"]: return False
    rp = c["reframe"]["passed"]; dp = c["delta"]["passed"]
    if not (rp or dp): return False
    if rp and not dp and c["delta"].get("median_dist", 0) < 0.20: return False
    if dp and not rp and c["reframe"].get("median_dist", 0) < 0.25: return False
    return True

def combine_dropped_anti(c):
    """Drop the κ=0.14 unreliable S_anti component; rest same."""
    if not c["wisdom"]["passed"]: return False
    rp = c["reframe"]["passed"]; dp = c["delta"]["passed"]
    if not (rp or dp): return False
    if rp and not dp and c["delta"].get("median_dist", 0) < 0.20: return False
    if dp and not rp and c["reframe"].get("median_dist", 0) < 0.25: return False
    return True


def run_gate(labels, combine_fn):
    results = {}
    for cid in list_candidates():
        comps = {
            "reframe": eval_reframe(cid, labels),
            "delta":   eval_delta(cid, labels),
            "wisdom":  eval_wisdom(cid, labels),
            "anti":    eval_anti(cid, labels),
        }
        results[cid] = {"components": comps, "overall_pass": combine_fn(comps)}
    return results


def pr_f1(pred, gt, all_cids):
    pred = set(pred) & set(all_cids); gt = set(gt) & set(all_cids)
    tp = len(pred & gt); fp = len(pred - gt); fn = len(gt - pred)
    p = tp/(tp+fp) if (tp+fp) else 0
    r = tp/(tp+fn) if (tp+fn) else 0
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if (p+r) else 0,
            "n_pred": len(pred)}


def main():
    # Load majority-vote labels from Exp 25
    mv_data = json.loads((AUTO_DIR / "exp25_multifamily_labels.json").read_text())
    mv_labels = mv_data["majority_vote"]

    # Also load single-family (Claude) labels from Exp 24 for comparison
    cl_labels = json.loads((AUTO_DIR / "exp24_subset_labels.json").read_text())

    print(f"Majority-vote labels: {len(mv_labels)} pids")
    print(f"Claude-only labels: {len(cl_labels)} pids\n")

    results_full_cl = run_gate(cl_labels, combine_full)
    results_full_mv = run_gate(mv_labels, combine_full)
    results_drop_mv = run_gate(mv_labels, combine_dropped_anti)

    gt = {"WCROSSL01", "WCAND07"}  # Exp 11 pseudo-ground-truth
    all_cids = list_candidates()

    exp17 = json.loads((AUTO_DIR / "exp17_trigger_conditioned_log.json").read_text())[-1]["results"]
    exp17_pass = {r["cid"] for r in exp17 if r.get("gate_pass")}

    configs = [
        ("exp17_researcher", exp17_pass),
        ("exp24_claude_only_full", {cid for cid, r in results_full_cl.items() if r["overall_pass"]}),
        ("exp28A_majvote_full",    {cid for cid, r in results_full_mv.items() if r["overall_pass"]}),
        ("exp28B_majvote_drop_anti", {cid for cid, r in results_drop_mv.items() if r["overall_pass"]}),
    ]

    print(f"{'gate':30s} {'n_pred':8s} {'precision':10s} {'recall':8s} {'f1':6s}  PASS set")
    print("-" * 100)
    out = {}
    for name, pred_set in configs:
        m = pr_f1(pred_set, gt, all_cids)
        out[name] = {"metrics": m, "pass_set": sorted(pred_set)}
        print(f"  {name:28s} {m['n_pred']:<8d} {m['precision']:<10.2f} "
              f"{m['recall']:<8.2f} {m['f1']:<6.2f}  {sorted(pred_set)}")

    print(f"\n  Ground truth (Exp 11 majority-of-3 at n=50): {sorted(gt)}")

    # Save
    OUT_LOG.write_text(json.dumps({
        "majority_vote_stats": {
            sub: sum(1 for lab in mv_labels.values() if lab.get(sub))
            for sub in ("S_reframe", "S_delta", "S_wisdom", "S_anti")
        },
        "gates": out,
        "ground_truth": sorted(gt),
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
