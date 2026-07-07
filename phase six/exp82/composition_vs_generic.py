"""Exp 82 v2 final gate: gated-EXT vs gated-GENERIC.

The 7 cross-judge winners from composition_analysis.py established that:
  gated_ext_correct = ext_correct if feature_fires else base_correct
keeps trigger Δ ≥ +5pp across 3 judges while forcing outside Δ ≈ 0.

But the same router architecture could be applied to GENERIC ('be careful'):
  gated_gen_correct = gen_correct if feature_fires else base_correct

If gated_ext > gated_gen across judges, the wisdom CONTENT (the kind's
structural injection) is adding value beyond what the router architecture
alone provides. If gated_ext ≈ gated_gen, the router architecture is the
real driver — wisdom content is irrelevant.

This is post-hoc, free, definitive.

Output: composition_vs_generic_summary.json
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
OUT = EXP_DIR / "composition_vs_generic_summary.json"


JUDGES = ("gpt_mini_strict", "gemini_strict", "claude_haiku_strict")
KINDS_TO_GATE = ("constraint", "decomposition", "verification", "hp_change")


def load_strict_grades() -> dict:
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
    counts: dict = defaultdict(int)
    seen: dict = defaultdict(set)
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
            if "gpt" in model.lower():
                clf = "gpt_mini"
            elif "gemini" in model.lower():
                clf = "gemini"
            elif "claude" in model.lower() or "haiku" in model.lower():
                clf = "claude_haiku"
            else:
                clf = "unknown"
            seen[(cid, pid)].add(clf)
            if fired:
                counts[(cid, pid)] += 1
    consensus = {}
    for k in seen:
        consensus[k] = (counts.get(k, 0) >= 2)
    return consensus


def main():
    print("Loading...", flush=True)
    grades = load_strict_grades()
    consensus = load_feature_consensus()
    hypos = [json.loads(l) for l in open(HYPO_PATH) if l.strip()]
    print(f"  grades: {len(grades)}, consensus: {len(consensus)}, hypos: {len(hypos)}", flush=True)

    # For each hypothesis × judge, compute gated EXT, gated GEN, uniform EXT, uniform GEN
    per_hyp = []
    by_kind_judge = defaultdict(list)
    for h in hypos:
        cid = h["seed_cid"]
        kind = h["kind"]
        if kind not in KINDS_TO_GATE:
            continue
        hid = h["hid"]
        trig = h["trigger_subset"]
        outs = h["outside_subset"]

        for judge in JUDGES:
            base_t = sum(grades.get((judge, hid, pid, "base"), 0) for pid in trig)
            ext_t = sum(grades.get((judge, hid, pid, "ext"), 0) for pid in trig)
            gen_t = sum(grades.get((judge, hid, pid, "generic"), 0) for pid in trig)
            base_o = sum(grades.get((judge, hid, pid, "base"), 0) for pid in outs)
            ext_o = sum(grades.get((judge, hid, pid, "ext"), 0) for pid in outs)
            gen_o = sum(grades.get((judge, hid, pid, "generic"), 0) for pid in outs)
            n_t = len(trig); n_o = len(outs)

            gated_ext_t = gated_gen_t = 0
            gated_ext_o = gated_gen_o = 0
            n_fire_t = n_fire_o = 0
            for pid in trig:
                fire = consensus.get((cid, pid), False)
                if fire:
                    n_fire_t += 1
                    gated_ext_t += grades.get((judge, hid, pid, "ext"), 0)
                    gated_gen_t += grades.get((judge, hid, pid, "generic"), 0)
                else:
                    g_base = grades.get((judge, hid, pid, "base"), 0)
                    gated_ext_t += g_base
                    gated_gen_t += g_base
            for pid in outs:
                fire = consensus.get((cid, pid), False)
                if fire:
                    n_fire_o += 1
                    gated_ext_o += grades.get((judge, hid, pid, "ext"), 0)
                    gated_gen_o += grades.get((judge, hid, pid, "generic"), 0)
                else:
                    g_base = grades.get((judge, hid, pid, "base"), 0)
                    gated_ext_o += g_base
                    gated_gen_o += g_base

            row = {
                "cid": cid, "kind": kind, "hid": hid, "judge": judge,
                "n_trigger": n_t, "n_outside": n_o,
                "n_fire_trigger": n_fire_t, "n_fire_outside": n_fire_o,
                # raw correctness counts
                "base_t": base_t, "ext_t": ext_t, "gen_t": gen_t,
                "gated_ext_t": gated_ext_t, "gated_gen_t": gated_gen_t,
                # deltas
                "delta_ext_base_uniform": (ext_t - base_t) / n_t if n_t else 0,
                "delta_ext_base_gated": (gated_ext_t - base_t) / n_t if n_t else 0,
                "delta_gen_base_uniform": (gen_t - base_t) / n_t if n_t else 0,
                "delta_gen_base_gated": (gated_gen_t - base_t) / n_t if n_t else 0,
                # THE KEY: gated_ext vs gated_gen
                "delta_ext_gen_uniform": (ext_t - gen_t) / n_t if n_t else 0,
                "delta_ext_gen_gated": (gated_ext_t - gated_gen_t) / n_t if n_t else 0,
            }
            per_hyp.append(row)
            by_kind_judge[(kind, judge)].append(row)

    # ── Per-kind aggregate ──────────────────────────────────────────
    print(f"\n=== Per-kind: gated-EXT vs gated-GEN (9 wisdoms × 3 judges) ===\n", flush=True)
    print(f"  {'kind':14s} | {'gat E-B':>9s} | {'gat G-B':>9s} | {'gat E-G':>9s} | {'uni E-G':>9s}", flush=True)
    print(f"  {'-'*14}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}-+-{'-'*9}", flush=True)
    summary = {}
    for kind in KINDS_TO_GATE:
        rows = []
        for judge in JUDGES:
            rows.extend(by_kind_judge.get((kind, judge), []))
        if not rows:
            continue
        g_eb = sum(r["delta_ext_base_gated"] for r in rows) / len(rows)
        g_gb = sum(r["delta_gen_base_gated"] for r in rows) / len(rows)
        g_eg = sum(r["delta_ext_gen_gated"] for r in rows) / len(rows)
        u_eg = sum(r["delta_ext_gen_uniform"] for r in rows) / len(rows)
        summary[kind] = {
            "gated_delta_ext_base": g_eb,
            "gated_delta_gen_base": g_gb,
            "gated_delta_ext_gen": g_eg,
            "uniform_delta_ext_gen": u_eg,
        }
        print(f"  {kind:14s} | {g_eb:+8.2%} | {g_gb:+8.2%} | {g_eg:+8.2%} | {u_eg:+8.2%}", flush=True)

    # ── Per-judge aggregate ─────────────────────────────────────────
    print(f"\n=== Per-judge: gated-EXT vs gated-GEN (avg across kinds × wisdoms) ===\n", flush=True)
    print(f"  {'judge':14s} | {'gated E-G':>9s} | {'uniform E-G':>11s}", flush=True)
    print(f"  {'-'*14}-+-{'-'*9}-+-{'-'*11}", flush=True)
    per_judge_summary = {}
    for judge in JUDGES:
        rows = []
        for kind in KINDS_TO_GATE:
            rows.extend(by_kind_judge.get((kind, judge), []))
        if not rows:
            continue
        g_eg = sum(r["delta_ext_gen_gated"] for r in rows) / len(rows)
        u_eg = sum(r["delta_ext_gen_uniform"] for r in rows) / len(rows)
        per_judge_summary[judge] = {"gated_ext_gen": g_eg, "uniform_ext_gen": u_eg}
        print(f"  {judge:14s} | {g_eg:+8.2%} | {u_eg:+10.2%}", flush=True)

    # ── The 7 winners from composition_analysis: do they still beat gated-GEN? ──
    winners_from_path2 = [
        ("WCAND01", "constraint"), ("WCAND01", "decomposition"),
        ("WCAND01", "verification"), ("WCAND01", "hp_change"),
        ("WCAND02", "hp_change"), ("WCAND03", "constraint"),
        ("WCAND03", "hp_change"),
    ]
    print(f"\n=== Path 2 winners: gated-EXT vs gated-GEN (per judge) ===\n", flush=True)
    print(f"  {'wisdom/kind':28s} | {'gpt_mini':>11s} | {'gemini':>11s} | {'haiku':>11s} | {'all3≥0?':>7s} | {'all3≥+5?':>9s}", flush=True)
    print(f"  {'-'*28}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}-+-{'-'*7}-+-{'-'*9}", flush=True)
    by_hyp_jrow = defaultdict(dict)
    for r in per_hyp:
        by_hyp_jrow[(r["cid"], r["kind"])][r["judge"]] = r

    survivors_zero = []
    survivors_strict = []
    for cid, kind in winners_from_path2:
        rows = by_hyp_jrow.get((cid, kind), {})
        if len(rows) != 3:
            continue
        d_eg = [rows[j]["delta_ext_gen_gated"] for j in JUDGES]
        all_ge0 = all(d >= 0 for d in d_eg)
        all_ge5 = all(d >= 0.05 for d in d_eg)
        if all_ge0: survivors_zero.append((cid, kind))
        if all_ge5: survivors_strict.append((cid, kind))
        mark_ge0 = "  ✓" if all_ge0 else "   "
        mark_ge5 = "    ✓" if all_ge5 else "     "
        print(f"  {cid+'/'+kind:28s} | {d_eg[0]:+10.2%} | {d_eg[1]:+10.2%} | {d_eg[2]:+10.2%} | {mark_ge0:>7s} | {mark_ge5:>9s}", flush=True)

    print(f"\n  → survivors at all3 (E-G ≥ 0): {len(survivors_zero)}/{len(winners_from_path2)}", flush=True)
    for s in survivors_zero:
        print(f"      {s[0]}/{s[1]}", flush=True)
    print(f"  → survivors at all3 strict (E-G ≥ +5%): {len(survivors_strict)}/{len(winners_from_path2)}", flush=True)
    for s in survivors_strict:
        print(f"      {s[0]}/{s[1]}", flush=True)

    # ── Per hypothesis full table: is wisdom content beating generic under router? ──
    print(f"\n=== ALL 36 hypotheses: ext-gen Δ under gating (cross-judge consensus) ===\n", flush=True)
    print(f"  {'wisdom/kind':28s} | {'gpt_mini':>11s} | {'gemini':>11s} | {'haiku':>11s} | {'min':>7s}", flush=True)
    print(f"  {'-'*28}-+-{'-'*11}-+-{'-'*11}-+-{'-'*11}-+-{'-'*7}", flush=True)
    by_kind_count = defaultdict(lambda: [0, 0])
    all_ge0_winners = []
    for (cid, kind), rows in sorted(by_hyp_jrow.items()):
        if len(rows) != 3:
            continue
        d_eg = [rows[j]["delta_ext_gen_gated"] for j in JUDGES]
        m = min(d_eg)
        all_ge0 = all(d >= 0 for d in d_eg)
        if all_ge0:
            all_ge0_winners.append((cid, kind))
            by_kind_count[kind][0] += 1
        by_kind_count[kind][1] += 1
        flag = " ✓" if all_ge0 else "  "
        print(f"  {cid+'/'+kind:28s} | {d_eg[0]:+10.2%} | {d_eg[1]:+10.2%} | {d_eg[2]:+10.2%} | {m:+6.2%}{flag}", flush=True)

    print(f"\n  → all 36 hypotheses' gated EXT≥gated GEN across all 3 judges: {len(all_ge0_winners)}", flush=True)
    print(f"\n  Per kind:", flush=True)
    for kind, c in by_kind_count.items():
        print(f"    {kind:14s}: {c[0]}/{c[1]}", flush=True)

    OUT.write_text(json.dumps({
        "per_hypothesis": per_hyp,
        "summary_per_kind": summary,
        "summary_per_judge": per_judge_summary,
        "path2_winners_survivors_ge0": survivors_zero,
        "path2_winners_survivors_strict": survivors_strict,
        "all_ge0_winners_full_36": all_ge0_winners,
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT.relative_to(PROJECT)}", flush=True)


if __name__ == "__main__":
    main()
