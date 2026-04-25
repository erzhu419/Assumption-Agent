"""Exp 49 — Hierarchical Bayesian re-analysis of audit-stack verdicts.

Replaces threshold-crossing rhetoric (FLIP / COLLAPSE / REPLICATE)
with continuous posterior probabilities P(true wr_ext > t | data) per
(KEEP, judge family) pair. Addresses R8 reviewer concern that the
paper "frequently uses point-estimate language ... even when intervals
overlap substantially."

Model. For each (KEEP, judge family, evaluation set) cell, observed
ext-wins out of non-tie pairs is Binomial(n_eff, theta). Place a
weakly-informative Beta(2, 2) prior on theta. Posterior is
Beta(2 + ext_wins, 2 + n_eff - ext_wins). Report posterior mean,
95% credible interval, and P(theta > 0.50), P(theta > 0.55),
P(theta > 0.60).

Sources of verdicts on identical answer pairs (cached sample_extend_50):
- gemini-3-flash (inner-loop judge) — Exp 10 inner verdict
- claude-haiku, gpt-mini — Exp 36 cheap re-judgment (cached pairs)
- claude-opus-4-6, gpt-5.4 — Exp 35 expensive re-judgment (cached pairs)

Sources on fresh data (Exp 47, 30 fresh pids, original 12 candidates):
- gemini-3-flash inner gate
- claude-haiku L1

Output: posterior table per (KEEP, judge_family, eval_set), summary
table of P(theta > 0.55) and P(theta > 0.60) per cell, plus aggregate
posterior across families per KEEP.

No new API calls. Pure analysis on existing verdict logs.
"""
import json
import math
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).parent.parent
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp49_hierarchical_bayes_log.json"

KEEPS = ["W076", "W077", "W078"]


def beta_logbeta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_cdf(t, a, b):
    """Regularised incomplete beta function I_t(a, b) — continued fraction."""
    if t <= 0: return 0.0
    if t >= 1: return 1.0
    bt = math.exp(-beta_logbeta(a, b) + a * math.log(t) + b * math.log(1 - t))
    if t < (a + 1) / (a + b + 2):
        return bt * _betacf(t, a, b) / a
    return 1.0 - bt * _betacf(1 - t, b, a) / b


def _betacf(x, a, b, max_iter=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < eps: break
    return h


def beta_quantile(p, a, b, n_steps=10000):
    """Bisection on beta_cdf to invert at probability p."""
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def posterior_summary(ext_wins, n_eff, prior_a=2.0, prior_b=2.0):
    a = prior_a + ext_wins
    b = prior_b + (n_eff - ext_wins)
    mean = a / (a + b)
    ci_lo = beta_quantile(0.025, a, b)
    ci_hi = beta_quantile(0.975, a, b)
    p_gt_50 = 1.0 - beta_cdf(0.50, a, b)
    p_gt_55 = 1.0 - beta_cdf(0.55, a, b)
    p_gt_60 = 1.0 - beta_cdf(0.60, a, b)
    return {
        "ext_wins": ext_wins, "n_eff": n_eff,
        "post_mean": round(mean, 3),
        "ci95": [round(ci_lo, 3), round(ci_hi, 3)],
        "P_gt_0.50": round(p_gt_50, 3),
        "P_gt_0.55": round(p_gt_55, 3),
        "P_gt_0.60": round(p_gt_60, 3),
    }


def tally_verdicts(verdicts):
    """Return (ext_wins, base_wins, ties, total)."""
    ext = base = tie = 0
    for v in verdicts.values() if isinstance(verdicts, dict) else verdicts:
        if v == "ext": ext += 1
        elif v == "base": base += 1
        elif v == "tie": tie += 1
    return ext, base, tie, ext + base + tie


def load_cached_verdicts():
    """Return {KEEP: {family: (ext_wins, n_eff)}} on cached extend_50 pairs."""
    out = {k: {} for k in KEEPS}
    e35 = json.loads((AUTO / "exp35_expensive_extended_log.json").read_text())
    for r in e35["results"]:
        kid = r["cand_id"]
        if kid not in out: continue
        for fam, v in r["verdicts"].items():
            ext, base, tie, total = tally_verdicts(v)
            out[kid][fam] = (ext, ext + base)
    e36 = json.loads((AUTO / "exp36_cheap_verdicts_log.json").read_text())
    for r in e36["results"]:
        kid = r["cand_id"]
        if kid not in out: continue
        for fam, v in r["verdicts"].items():
            ext, base, tie, total = tally_verdicts(v)
            out[kid][fam] = (ext, ext + base)
    return out


def load_fresh_verdicts():
    """Return {KEEP: {family: (ext_wins, n_eff)}} from Exp 47 fresh-data.
    Exp 47 stores per-candidate ext/base/tie counts in `summary` list."""
    out = {k: {} for k in KEEPS}
    log = json.loads((AUTO / "exp47_preregistered_fresh_loop_log.json").read_text())
    cand2keep = {"WCAND05": "W076", "WCAND10": "W077", "WCROSSL01": "W078"}
    for r in log.get("summary", []):
        cand = r.get("tid")
        if cand not in cand2keep: continue
        kid = cand2keep[cand]
        ext, base, n_eff = r.get("ext", 0), r.get("base", 0), r.get("n_eff", 0)
        if n_eff > 0:
            out[kid]["gemini_inner_fresh"] = (ext, n_eff)
        # L1 cached as separate ext/base counts; the log stores the wr only.
        # Reconstruct from wr_l1 * n_eff_l1 (rounded). Inner ties already excluded.
        wr_l1 = r.get("wr_l1")
        n_l1 = r.get("n_eff_l1", 0)
        if wr_l1 is not None and n_l1 > 0:
            ext_l1 = round(wr_l1 * n_l1)
            out[kid]["claude_haiku_l1_fresh"] = (ext_l1, n_l1)
    return out


def aggregate_posterior(family_data, prior_a=2.0, prior_b=2.0):
    """Pool data across all families for the same KEEP — posterior under
    independent-judge assumption. Returns same dict shape as posterior_summary."""
    total_wins = sum(w for w, n in family_data.values())
    total_n = sum(n for w, n in family_data.values())
    return posterior_summary(total_wins, total_n, prior_a, prior_b)


def main():
    cached = load_cached_verdicts()
    fresh = load_fresh_verdicts()

    print(f"Cached extend_50 verdicts loaded: "
          f"{[(k, list(cached[k].keys())) for k in KEEPS]}\n")
    print(f"Fresh-data Exp 47 verdicts loaded: "
          f"{[(k, list(fresh[k].keys())) for k in KEEPS]}\n")

    summary = {}
    print("=== Per (KEEP, judge family, eval set) posterior ===")
    print(f"{'KEEP':5s} {'family':22s} {'set':7s} {'wins/n':>10s} "
          f"{'post_mean':>10s} {'CI95':>16s} "
          f"{'P>.50':>7s} {'P>.55':>7s} {'P>.60':>7s}")
    print("-" * 102)
    for kid in KEEPS:
        summary[kid] = {"per_family": {}, "aggregate": {}}
        for fam, (w, n) in cached[kid].items():
            ps = posterior_summary(w, n)
            print(f"{kid:5s} {fam:22s} {'cached':7s} {f'{w}/{n}':>10s} "
                  f"{ps['post_mean']:>10.3f} {str(ps['ci95']):>16s} "
                  f"{ps['P_gt_0.50']:>7.3f} {ps['P_gt_0.55']:>7.3f} "
                  f"{ps['P_gt_0.60']:>7.3f}")
            summary[kid]["per_family"][f"{fam}_cached"] = ps
        for fam, (w, n) in fresh[kid].items():
            ps = posterior_summary(w, n)
            print(f"{kid:5s} {fam:22s} {'fresh':7s} {f'{w}/{n}':>10s} "
                  f"{ps['post_mean']:>10.3f} {str(ps['ci95']):>16s} "
                  f"{ps['P_gt_0.50']:>7.3f} {ps['P_gt_0.55']:>7.3f} "
                  f"{ps['P_gt_0.60']:>7.3f}")
            summary[kid]["per_family"][fam] = ps

    print(f"\n=== Aggregate posterior across all families (cached only) ===")
    print(f"{'KEEP':5s} {'wins/n':>10s} {'post_mean':>10s} {'CI95':>16s} "
          f"{'P>.50':>7s} {'P>.55':>7s} {'P>.60':>7s}")
    print("-" * 80)
    for kid in KEEPS:
        ps = aggregate_posterior(cached[kid])
        wn = "{}/{}".format(ps["ext_wins"], ps["n_eff"])
        print(f"{kid:5s} {wn:>10s} "
              f"{ps['post_mean']:>10.3f} {str(ps['ci95']):>16s} "
              f"{ps['P_gt_0.50']:>7.3f} {ps['P_gt_0.55']:>7.3f} "
              f"{ps['P_gt_0.60']:>7.3f}")
        summary[kid]["aggregate"]["cached_pooled"] = ps

    print(f"\n=== Combined cached + fresh posterior ===")
    for kid in KEEPS:
        combined = dict(cached[kid])
        combined.update(fresh[kid])
        ps = aggregate_posterior(combined)
        wn = "{}/{}".format(ps["ext_wins"], ps["n_eff"])
        print(f"{kid:5s} {wn:>10s} "
              f"{ps['post_mean']:>10.3f} {str(ps['ci95']):>16s} "
              f"{ps['P_gt_0.50']:>7.3f} {ps['P_gt_0.55']:>7.3f} "
              f"{ps['P_gt_0.60']:>7.3f}")
        summary[kid]["aggregate"]["all_pooled"] = ps

    print(f"\n=== Headline ===")
    print("Posterior P(theta > 0.55) on the cached pooled data:")
    for kid in KEEPS:
        p = summary[kid]["aggregate"]["cached_pooled"]["P_gt_0.55"]
        verdict = "STRONG (>0.95)" if p > 0.95 else \
                  "MODERATE (0.5–0.95)" if p > 0.5 else \
                  "WEAK (<0.5)"
        print(f"  {kid}: P(theta > 0.55) = {p:.3f}  → {verdict}")
    print()
    print("Posterior P(theta > 0.60) on the cached pooled data:")
    for kid in KEEPS:
        p = summary[kid]["aggregate"]["cached_pooled"]["P_gt_0.60"]
        verdict = "STRONG (>0.95)" if p > 0.95 else \
                  "MODERATE (0.5–0.95)" if p > 0.5 else \
                  "WEAK (<0.5)"
        print(f"  {kid}: P(theta > 0.60) = {p:.3f}  → {verdict}")

    OUT_LOG.write_text(json.dumps({
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prior": "Beta(2, 2)",
        "summary": summary,
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
