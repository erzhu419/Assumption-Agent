"""Exp 48 — gpt-5.5 as 'human ground truth' substitute: per-judge
calibration.

For each pid where gpt-5.5 produced a structured 4-dim Likert
rating (Exp 38: W078 × 30 pids; Exp 41: W076 × 30, W077 × 30 = 90
total ratings), compute gpt-5.5's combined preference (ext > base
/ base > ext / tie) per pid. Then for the same (KEEP, pid) pair,
look up each panel judge's pairwise verdict (from Exp 35
expensive + Exp 36 cheap re-runs).

Per-judge agreement with gpt-5.5: fraction of pids where the
judge's verdict agrees with gpt-5.5's preference (treating gpt-5.5
'tie' as agreeing with judge 'tie' or with neither directionally).

If a panel judge has LOW agreement with gpt-5.5, the implication
is that judge's verdicts diverge from a stronger non-panel rater
on the same content. If gemini-3-flash (the inner-loop judge)
has the LOWEST agreement with gpt-5.5, the original gate's
positive verdict is unsupported by stronger judges.

This is post-hoc analysis on existing logs; no new API calls.
"""

import json
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).parent.parent
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp48_gpt55_calibration_log.json"


def load_gpt55_ratings():
    """Return {keep_id: {pid: 'ext_better' / 'base_better' / 'tie'}}."""
    out = {}

    # Exp 38: W078 only
    e38 = json.loads((AUTO / "exp38_gpt55_structured_audit_log.json").read_text())
    out["W078"] = {}
    for r in e38["results"]:
        if "error" in r:
            continue
        es = r.get("ext_scores", {}); bs = r.get("base_scores", {})
        if not es or not bs:
            continue
        try:
            ext_avg = sum(es[d] for d in ["substantive", "method", "fit", "prescriptive"]) / 4
            base_avg = sum(bs[d] for d in ["substantive", "method", "fit", "prescriptive"]) / 4
            if ext_avg > base_avg + 0.1:
                out["W078"][r["pid"]] = "ext_better"
            elif base_avg > ext_avg + 0.1:
                out["W078"][r["pid"]] = "base_better"
            else:
                out["W078"][r["pid"]] = "tie"
        except KeyError:
            continue

    # Exp 41: W076, W077
    e41 = json.loads((AUTO / "exp41_gpt55_rating_all_keeps_log.json").read_text())
    for kp in e41["results_per_keep"]:
        kid = kp["keep_id"]
        out[kid] = {}
        for r in kp["results"]:
            if "error" in r:
                continue
            es = r.get("ext_scores", {}); bs = r.get("base_scores", {})
            if not es or not bs:
                continue
            try:
                ext_avg = sum(es[d] for d in ["substantive", "method", "fit", "prescriptive"]) / 4
                base_avg = sum(bs[d] for d in ["substantive", "method", "fit", "prescriptive"]) / 4
                if ext_avg > base_avg + 0.1:
                    out[kid][r["pid"]] = "ext_better"
                elif base_avg > ext_avg + 0.1:
                    out[kid][r["pid"]] = "base_better"
                else:
                    out[kid][r["pid"]] = "tie"
            except KeyError:
                continue

    return out


def load_panel_verdicts():
    """Return {keep_id: {family: {pid: 'ext'/'base'/'tie'}}}."""
    out = {}
    e35 = json.loads((AUTO / "exp35_expensive_extended_log.json").read_text())
    for r in e35["results"]:
        kid = r["cand_id"]
        out.setdefault(kid, {})
        for fam, v in r["verdicts"].items():
            out[kid].setdefault(fam, {}).update(v)

    e36 = json.loads((AUTO / "exp36_cheap_verdicts_log.json").read_text())
    for r in e36["results"]:
        kid = r["cand_id"]
        out.setdefault(kid, {})
        for fam, v in r["verdicts"].items():
            out[kid].setdefault(fam, {}).update(v)

    return out


def map_panel_to_gpt55(panel_v):
    """Map panel verdict 'ext'/'base'/'tie' to {ext_better, base_better, tie}."""
    if panel_v == "ext": return "ext_better"
    if panel_v == "base": return "base_better"
    if panel_v == "tie": return "tie"
    return None  # err / missing


def main():
    gpt55 = load_gpt55_ratings()
    panel = load_panel_verdicts()
    print(f"gpt-5.5 ratings loaded: {[(k, len(v)) for k, v in gpt55.items()]}")
    print(f"panel verdicts loaded: {[(k, list(v.keys())) for k, v in panel.items()]}\n")

    # For each (keep_id, family), compute agreement-rate with gpt-5.5
    families = ["gemini", "claude_haiku", "gpt_mini", "claude_opus", "gpt5"]
    print(f"{'keep_id':10s}  {'family':14s}  {'agree':>8s}  {'n':>4s}  per-pid breakdown")
    print("-" * 90)

    summary = {}
    for kid in sorted(gpt55):
        if kid not in panel: continue
        for fam in families:
            if fam not in panel[kid]: continue
            agree = 0
            total = 0
            tally = defaultdict(int)
            for pid, gpt55_pref in gpt55[kid].items():
                panel_v = panel[kid][fam].get(pid)
                if panel_v is None: continue
                mapped = map_panel_to_gpt55(panel_v)
                if mapped is None: continue
                total += 1
                if mapped == gpt55_pref:
                    agree += 1
                    tally["agree"] += 1
                else:
                    tally[f"disagree:{gpt55_pref}->{mapped}"] += 1
            agree_rate = agree / total if total else 0
            print(f"  {kid:10s}  {fam:14s}  {agree_rate:>8.2f}  {total:>4d}  "
                  f"agree={agree} disagree={total-agree}")
            summary.setdefault(kid, {})[fam] = {
                "agreement_rate": agree_rate, "n": total, "n_agree": agree,
            }

    # Aggregate per-family across 3 KEEPs
    print(f"\n=== Aggregate agreement per panel judge with gpt-5.5 ('human-substitute') ===")
    print(f"{'family':16s}  {'mean_agree':>11s}  {'total_n':>9s}  per-keep")
    print("-" * 80)
    for fam in families:
        rates = [summary[k][fam]["agreement_rate"]
                  for k in summary if fam in summary[k]]
        ns = [summary[k][fam]["n"] for k in summary if fam in summary[k]]
        agreesum = sum(summary[k][fam]["n_agree"] for k in summary if fam in summary[k])
        n_total = sum(ns)
        if rates:
            mean = sum(rates) / len(rates)
            line = f"  {fam:16s}  {mean:>11.2f}  {n_total:>9d}  "
            line += " | ".join(f"{k}={summary[k][fam]['agreement_rate']:.2f}"
                               for k in summary if fam in summary[k])
            print(line)

    # Headline
    print(f"\n=== Headline interpretation ===")
    fam_means = {}
    for fam in families:
        rates = [summary[k][fam]["agreement_rate"]
                  for k in summary if fam in summary[k]]
        if rates:
            fam_means[fam] = sum(rates) / len(rates)

    if fam_means:
        sorted_fams = sorted(fam_means.items(), key=lambda x: -x[1])
        print(f"  Highest agreement with gpt-5.5: {sorted_fams[0][0]} ({sorted_fams[0][1]:.2f})")
        print(f"  Lowest agreement with gpt-5.5: {sorted_fams[-1][0]} ({sorted_fams[-1][1]:.2f})")
        print()
        gemini_rate = fam_means.get("gemini", None)
        if gemini_rate is not None:
            other_rates = {f: r for f, r in fam_means.items() if f != "gemini"}
            other_mean = sum(other_rates.values()) / len(other_rates)
            diff = gemini_rate - other_mean
            print(f"  gemini-3-flash (inner-loop judge): {gemini_rate:.2f}")
            print(f"  Other panel judges mean: {other_mean:.2f}")
            print(f"  gemini - other_mean: {diff:+.2f}")
            if diff < -0.05:
                print(f"  Reading: gemini agrees with gpt-5.5 LESS than other judges --- ")
                print(f"  the inner-loop's same-family verdict is the FURTHEST from a stronger")
                print(f"  non-panel rater. Original gate likely captured gemini-specific preference.")
            elif diff > 0.05:
                print(f"  Reading: gemini agrees MORE with gpt-5.5 than other judges --- ")
                print(f"  the inner-loop's same-family verdict is closest to gpt-5.5's view.")
            else:
                print(f"  Reading: gemini's gpt-5.5 agreement is comparable to other judges.")

    out = {"timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"),
           "summary": summary, "family_means": fam_means}
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
