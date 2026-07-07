"""Exp 82 v2 step path-2: feature × structure GATING (post-hoc).

The cross-judge regrade gave us per-(hypothesis, pid, judge) base/ext/gen
correctness. The LLM-classifier feature run gave us per-(wisdom, pid)
3-classifier-consensus fire/no-fire. Compose:

  gated_ext_correct(cid, kind, pid, judge) =
      ext_correct(...)   if feature_consensus_fire(cid, pid) else
      base_correct(...)

Then per (kind, judge):
  - gated_delta = gated_correct − base_correct  on trigger_subset
                                                 on outside_subset
  - vs uniform: did gating improve over uniform-ext on outside (specificity)?
                did gating cost on trigger (some pids fall back to base)?

No new LLM calls. Pure post-hoc.

Output: composition_summary.json + console summary table.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent

EXP_DIR = Path(__file__).parent
HYPO_PATH = EXP_DIR / "hypotheses.jsonl"
REGRADE_FORENSIC = EXP_DIR / "regrade_strict_forensic.jsonl"
FEATURE_LOG = EXP_DIR / "feature_classify_log.jsonl"
OUT = EXP_DIR / "composition_summary.json"


JUDGES = ("gpt_mini_strict", "gemini_strict", "claude_haiku_strict")
KINDS_TO_GATE = ("constraint", "decomposition", "verification", "hp_change")


def load_strict_grades() -> dict:
    """Returns {(judge, hid, pid, condition): 0/1}."""
    out = {}
    with open(REGRADE_FORENSIC, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            judge = r.get("judge", "")
            hid = r.get("hid", "")
            pid = r.get("pid", "")
            cond = r.get("condition", "")
            res = r.get("result", 0)
            if not all([judge, hid, pid, cond]):
                continue
            out[(judge, hid, pid, cond)] = int(res)
    return out


def load_feature_consensus() -> dict:
    """3-classifier consensus per (cid, pid): ≥2 of 3 fired ⇒ True."""
    counts: dict = defaultdict(int)
    classifiers_seen: dict = defaultdict(set)
    with open(FEATURE_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pid = r.get("pid", "")
            cid = r.get("cid", "")
            fired = r.get("fired", 0)
            model = r.get("model", "")
            # Identify classifier family from model name
            if "gpt" in model.lower():
                clf = "gpt_mini"
            elif "gemini" in model.lower():
                clf = "gemini"
            elif "claude" in model.lower() or "haiku" in model.lower():
                clf = "claude_haiku"
            else:
                clf = "unknown"
            classifiers_seen[(cid, pid)].add(clf)
            if fired:
                counts[(cid, pid)] += 1
    consensus = {}
    for k, v in counts.items():
        consensus[k] = (v >= 2)
    # Make sure every (cid, pid) we saw has a consensus entry — even if 0 fires
    for k in classifiers_seen:
        consensus.setdefault(k, False)
    return consensus


def main():
    print("Loading strict grades + feature consensus + hypotheses...", flush=True)
    grades = load_strict_grades()
    consensus = load_feature_consensus()
    hypos = [json.loads(l) for l in open(HYPO_PATH) if l.strip()]
    print(f"  strict grades: {len(grades)} cells", flush=True)
    print(f"  feature consensus: {len(consensus)} (cid, pid) pairs", flush=True)
    print(f"  hypotheses: {len(hypos)}", flush=True)

    # For each (judge, hypothesis) compute uniform vs gated metrics
    print(f"\n{'-'*120}", flush=True)
    print(f"{'wisdom/kind':28s} | {'judge':14s} | {'uniform Δ trig':>14s} | {'gated Δ trig':>13s} | {'uniform Δ out':>14s} | {'gated Δ out':>12s} | {'fire rate trig':>14s}", flush=True)
    print(f"{'-'*120}", flush=True)

    by_kind_judge = defaultdict(list)  # (kind, judge) → list of dicts
    per_hyp = []

    for h in hypos:
        cid = h["seed_cid"]
        kind = h["kind"]
        if kind not in KINDS_TO_GATE:
            continue
        hid = h["hid"]
        trig = h["trigger_subset"]
        outs = h["outside_subset"]

        for judge in JUDGES:
            # Uniform EXT
            base_t = sum(grades.get((judge, hid, pid, "base"), 0) for pid in trig)
            ext_t = sum(grades.get((judge, hid, pid, "ext"), 0) for pid in trig)
            base_o = sum(grades.get((judge, hid, pid, "base"), 0) for pid in outs)
            ext_o = sum(grades.get((judge, hid, pid, "ext"), 0) for pid in outs)
            n_t = len(trig); n_o = len(outs)

            # Gated EXT — use ext when feature consensus fires, else base
            gated_t = 0
            gated_o = 0
            n_fire_t = 0
            n_fire_o = 0
            for pid in trig:
                fire = consensus.get((cid, pid), False)
                if fire:
                    gated_t += grades.get((judge, hid, pid, "ext"), 0)
                    n_fire_t += 1
                else:
                    gated_t += grades.get((judge, hid, pid, "base"), 0)
            for pid in outs:
                fire = consensus.get((cid, pid), False)
                if fire:
                    gated_o += grades.get((judge, hid, pid, "ext"), 0)
                    n_fire_o += 1
                else:
                    gated_o += grades.get((judge, hid, pid, "base"), 0)

            uni_d_t = (ext_t - base_t) / n_t if n_t else 0
            gat_d_t = (gated_t - base_t) / n_t if n_t else 0
            uni_d_o = (ext_o - base_o) / n_o if n_o else 0
            gat_d_o = (gated_o - base_o) / n_o if n_o else 0
            fire_rate_t = n_fire_t / n_t if n_t else 0
            fire_rate_o = n_fire_o / n_o if n_o else 0

            row = {
                "cid": cid, "kind": kind, "hid": hid, "judge": judge,
                "n_trigger": n_t, "n_outside": n_o,
                "fire_rate_trigger": fire_rate_t,
                "fire_rate_outside": fire_rate_o,
                "uniform_delta_trigger": uni_d_t,
                "gated_delta_trigger": gat_d_t,
                "uniform_delta_outside": uni_d_o,
                "gated_delta_outside": gat_d_o,
                "uniform_specificity": uni_d_t - uni_d_o,
                "gated_specificity": gat_d_t - gat_d_o,
            }
            by_kind_judge[(kind, judge)].append(row)
            per_hyp.append(row)
            short_judge = judge.replace("_strict", "")
            print(f"{cid+'/'+kind:28s} | {short_judge:14s} | {uni_d_t:+13.2%} | {gat_d_t:+12.2%} | {uni_d_o:+13.2%} | {gat_d_o:+11.2%} | {fire_rate_t:>13.0%}", flush=True)

    # Per-kind aggregate (mean across wisdoms × judges)
    print(f"\n{'='*100}", flush=True)
    print(f"\n=== Per-kind: mean Δ across 9 wisdoms × 3 judges ===\n", flush=True)
    print(f"  {'kind':14s} | {'uniform trig':>13s} | {'gated trig':>11s} | {'uniform out':>12s} | {'gated out':>10s} | {'uni spec':>9s} | {'gated spec':>11s}", flush=True)
    print(f"  {'-'*14}-+-{'-'*13}-+-{'-'*11}-+-{'-'*12}-+-{'-'*10}-+-{'-'*9}-+-{'-'*11}", flush=True)
    summary_per_kind = {}
    for kind in KINDS_TO_GATE:
        rows = []
        for judge in JUDGES:
            rows.extend(by_kind_judge.get((kind, judge), []))
        if not rows:
            continue
        avg_uni_t = sum(r["uniform_delta_trigger"] for r in rows) / len(rows)
        avg_gat_t = sum(r["gated_delta_trigger"] for r in rows) / len(rows)
        avg_uni_o = sum(r["uniform_delta_outside"] for r in rows) / len(rows)
        avg_gat_o = sum(r["gated_delta_outside"] for r in rows) / len(rows)
        uni_spec = avg_uni_t - avg_uni_o
        gat_spec = avg_gat_t - avg_gat_o
        summary_per_kind[kind] = {
            "n_rows": len(rows),
            "uniform_trigger": avg_uni_t,
            "gated_trigger": avg_gat_t,
            "uniform_outside": avg_uni_o,
            "gated_outside": avg_gat_o,
            "uniform_specificity": uni_spec,
            "gated_specificity": gat_spec,
        }
        print(f"  {kind:14s} | {avg_uni_t:+12.2%} | {avg_gat_t:+10.2%} | {avg_uni_o:+11.2%} | {avg_gat_o:+9.2%} | {uni_spec:+8.2%} | {gat_spec:+10.2%}", flush=True)

    # Per (kind × judge) — see if gating improves outside specificity in every judge
    print(f"\n=== Per kind × judge: does gating improve specificity? (gated_spec - uniform_spec) ===\n", flush=True)
    print(f"  {'kind':14s} | {'gpt_mini':>10s} | {'gemini':>10s} | {'haiku':>10s}", flush=True)
    print(f"  {'-'*14}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}", flush=True)
    for kind in KINDS_TO_GATE:
        cells = {}
        for judge in JUDGES:
            rows = by_kind_judge.get((kind, judge), [])
            if rows:
                avg_uni_t = sum(r["uniform_delta_trigger"] for r in rows) / len(rows)
                avg_gat_t = sum(r["gated_delta_trigger"] for r in rows) / len(rows)
                avg_uni_o = sum(r["uniform_delta_outside"] for r in rows) / len(rows)
                avg_gat_o = sum(r["gated_delta_outside"] for r in rows) / len(rows)
                d_spec = (avg_gat_t - avg_gat_o) - (avg_uni_t - avg_uni_o)
                cells[judge] = d_spec
            else:
                cells[judge] = 0
        gpt = cells.get("gpt_mini_strict", 0)
        gem = cells.get("gemini_strict", 0)
        haik = cells.get("claude_haiku_strict", 0)
        print(f"  {kind:14s} | {gpt:+9.2%} | {gem:+9.2%} | {haik:+9.2%}", flush=True)

    # Cross-judge winners: gated specificity > 5pp AND gated trigger Δ > 5pp under all 3 judges
    print(f"\n=== Cross-judge winners under GATING: per hypothesis ===\n", flush=True)
    by_hyp = defaultdict(dict)
    for r in per_hyp:
        by_hyp[(r["cid"], r["kind"])][r["judge"]] = r
    winners = []
    print(f"  {'wisdom/kind':28s} | {'gated trig (gpt/gem/hk)':>26s} | {'gated out (gpt/gem/hk)':>24s} | {'all3_trig≥5%':>13s} | {'all3_out≤5%':>12s}", flush=True)
    print(f"  {'-'*28}-+-{'-'*26}-+-{'-'*24}-+-{'-'*13}-+-{'-'*12}", flush=True)
    for (cid, kind), j_rows in by_hyp.items():
        g_trig = [j_rows[j]["gated_delta_trigger"] for j in JUDGES if j in j_rows]
        g_out  = [j_rows[j]["gated_delta_outside"] for j in JUDGES if j in j_rows]
        if len(g_trig) != 3:
            continue
        all3_trig = all(d >= 0.05 for d in g_trig)
        all3_out_lo = all(d <= 0.05 for d in g_out)
        mark = " ✓" if all3_trig and all3_out_lo else "  "
        if all3_trig and all3_out_lo:
            winners.append((cid, kind))
        print(f"  {cid+'/'+kind:28s} | {g_trig[0]:+8.0%}/{g_trig[1]:+8.0%}/{g_trig[2]:+8.0%} | {g_out[0]:+7.0%}/{g_out[1]:+7.0%}/{g_out[2]:+7.0%} | {'  ✓ ' if all3_trig else '    ':>13s} | {'  ✓ ' if all3_out_lo else '    ':>12s}{mark}", flush=True)

    print(f"\n  → cross-judge winners under GATING: {len(winners)}", flush=True)
    for cid, kind in winners:
        print(f"      {cid}/{kind}", flush=True)

    OUT.write_text(json.dumps({
        "per_hypothesis": per_hyp,
        "summary_per_kind": summary_per_kind,
        "winners_under_gating": winners,
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
