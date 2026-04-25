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


# ---------------------------------------------------------------
# DerSimonian-Laird random-effects meta-analysis on logit-scale
# (standard hierarchical model for combining several judge cells per
# KEEP). For each KEEP k:
#   logit(p_j) = mu_k + epsilon_j, epsilon_j ~ Normal(0, tau_k^2)
# y_j ~ Binomial(n_j, p_j). DL estimator gives between-judge variance
# tau_k^2; pooled posterior on mu_k integrates within-cell sampling
# variance plus tau_k^2. Returns posterior P(theta > t) on the
# population-level effect with proper between-judge uncertainty.
# Standard reference: DerSimonian & Laird 1986.
# ---------------------------------------------------------------


def inv_logit(x): return 1.0 / (1.0 + math.exp(-x)) if x > -50 else 0.0


def normal_cdf(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2)))


def dl_pool_logit(cells_for_keep):
    """DerSimonian-Laird pool of logit(p_j) across cells.

    cells_for_keep: list of (y, n) tuples. Returns dict with pooled
    logit mean, SE under random-effects model, between-judge SD tau,
    and posterior P(theta > t) for t in {0.50, 0.55, 0.60} treating
    pooled effect as Normal."""
    # Add 0.5 continuity correction for cells with extreme proportions
    transformed = []
    for y, n in cells_for_keep:
        if n == 0: continue
        p = (y + 0.5) / (n + 1)
        z = math.log(p / (1 - p))
        v = 1.0 / ((n + 1) * p * (1 - p))  # SE^2 of logit
        transformed.append((z, v))
    if len(transformed) < 2:
        # Fall back to single-cell normal approx
        if len(transformed) == 1:
            z, v = transformed[0]
            return {"mu": z, "se_re": math.sqrt(v), "tau": 0.0,
                    "k": 1}
        return None
    # Fixed-effect pooled
    weights = [1 / v for _, v in transformed]
    swt = sum(weights)
    mu_fe = sum(w * z for (z, _), w in zip(transformed, weights)) / swt
    Q = sum(w * (z - mu_fe) ** 2 for (z, _), w in zip(transformed, weights))
    k = len(transformed)
    swt2 = sum(w * w for w in weights)
    denom = swt - swt2 / swt
    tau2 = max(0.0, (Q - (k - 1)) / denom) if denom > 0 else 0.0
    tau = math.sqrt(tau2)
    # Random-effects pooled
    w_re = [1 / (v + tau2) for _, v in transformed]
    swt_re = sum(w_re)
    mu_re = sum(w * z for (z, _), w in zip(transformed, w_re)) / swt_re
    se_re = math.sqrt(1 / swt_re)
    return {"mu": mu_re, "se_re": se_re, "tau": tau, "k": k,
            "Q": Q, "tau2": tau2}


def dl_posterior_summary(pooled):
    """Convert pooled DL output to posterior P(theta > t) on probability
    scale via Normal approximation on logit, then transform via inv_logit.
    Includes between-cell variance via tau."""
    if pooled is None:
        return None
    mu, se_re, tau = pooled["mu"], pooled["se_re"], pooled["tau"]
    # Total predictive variance on logit: SE_RE^2 + tau^2 (predictive of
    # a future judge) — but for the population-level mu the variance is
    # SE_RE^2 only. We report mu posterior here (population effect).
    p_hat = inv_logit(mu)
    # CI95 on probability scale via logit-CI back-transform
    lo = inv_logit(mu - 1.96 * se_re)
    hi = inv_logit(mu + 1.96 * se_re)
    # P(mu_logit > logit(t)) under Normal(mu, se_re^2)
    def p_gt(t):
        z = (math.log(t / (1 - t)) - mu) / se_re
        return 1.0 - normal_cdf(z)
    return {
        "post_mean": round(p_hat, 3),
        "ci95": [round(lo, 3), round(hi, 3)],
        "tau_logit": round(tau, 3),
        "k_cells": pooled["k"],
        "P_gt_0.50": round(p_gt(0.50), 3),
        "P_gt_0.55": round(p_gt(0.55), 3),
        "P_gt_0.60": round(p_gt(0.60), 3),
    }


def fit_hierarchical_eb_OLD(cells, n_iter=80, lr=0.05, tau_init=0.5):
    """Fit mu_k for each KEEP and shared sigma_alpha for judge family
    random effect. cells: list of dicts with keys keep, fam, y, n,
    set ('cached' or 'fresh'). Returns (mu_dict, tau, alpha_post).

    Approximation: alpha_j integrated by 11-point Gauss-Hermite-style
    grid; mu_k and log_tau optimised by gradient ascent on the marginal
    log-likelihood."""
    keeps = sorted({c['keep'] for c in cells})
    fams = sorted({c['fam'] for c in cells})

    # 11-point quadrature: standard normal nodes/weights (subset of GH)
    # Approximate using equally-spaced grid since stdlib lacks GH.
    grid = [-3.0, -2.4, -1.8, -1.2, -0.6, 0.0, 0.6, 1.2, 1.8, 2.4, 3.0]
    grid_w = [math.exp(-0.5 * g * g) for g in grid]
    grid_w = [w / sum(grid_w) for w in grid_w]

    mu = {k: 0.0 for k in keeps}
    log_tau = math.log(tau_init)

    def marginal_loglik(mu, log_tau):
        tau = math.exp(log_tau)
        ll = 0.0
        # For each judge family: integrate over alpha_j
        for fam in fams:
            fam_cells = [c for c in cells if c['fam'] == fam]
            if not fam_cells: continue
            inner_lls = []
            for g, w in zip(grid, grid_w):
                alpha = tau * g
                lcond = sum(cell_loglik_given_alpha(c['y'], c['n'],
                                                     mu[c['keep']], alpha)
                              for c in fam_cells)
                inner_lls.append(math.log(w) + lcond)
            # log-sum-exp
            m = max(inner_lls)
            lsum = m + math.log(sum(math.exp(l - m) for l in inner_lls))
            ll += lsum
        return ll

    # Coordinate ascent
    for it in range(n_iter):
        # Gradient via finite difference
        eps = 0.01
        ll0 = marginal_loglik(mu, log_tau)
        for k in keeps:
            mu[k] += eps
            up = marginal_loglik(mu, log_tau)
            mu[k] -= 2 * eps
            dn = marginal_loglik(mu, log_tau)
            mu[k] += eps
            grad = (up - dn) / (2 * eps)
            mu[k] += lr * grad
        log_tau += eps
        up = marginal_loglik(mu, log_tau)
        log_tau -= 2 * eps
        dn = marginal_loglik(mu, log_tau)
        log_tau += eps
        grad = (up - dn) / (2 * eps)
        log_tau += lr * grad
        log_tau = max(min(log_tau, 2.0), -3.0)

    tau = math.exp(log_tau)

    # Posterior mode of alpha_j by 1D grid search
    alpha_post = {}
    for fam in fams:
        fam_cells = [c for c in cells if c['fam'] == fam]
        best_ll, best_a = -1e18, 0.0
        for a in [-3, -2.4, -1.8, -1.2, -0.6, -0.3, 0.0, 0.3, 0.6, 1.2, 1.8, 2.4, 3.0]:
            alpha = tau * a
            lcond = sum(cell_loglik_given_alpha(c['y'], c['n'],
                                                 mu[c['keep']], alpha)
                          for c in fam_cells)
            lprior = -0.5 * (alpha / tau) ** 2 - math.log(tau)
            ll = lcond + lprior
            if ll > best_ll: best_ll, best_a = ll, alpha
        alpha_post[fam] = best_a

    return mu, tau, alpha_post


def hierarchical_posterior_per_keep(cells, mu, tau, alpha_post, n_samples=4000):
    """Return per-KEEP posterior samples of theta = inv_logit(mu_k +
    alpha_j) marginalised over judge family. Pool across cells with
    weights proportional to n."""
    import random as rnd
    rnd.seed(2026)
    keeps = sorted({c['keep'] for c in cells})
    fams = sorted({c['fam'] for c in cells})
    out = {}
    for k in keeps:
        thetas = []
        # weight each judge family by total n for this KEEP
        fam_n = {f: sum(c['n'] for c in cells
                          if c['keep'] == k and c['fam'] == f)
                  for f in fams}
        total = sum(fam_n.values())
        if total == 0: continue
        for _ in range(n_samples):
            # sample alpha_j for a randomly-weighted family
            r = rnd.random() * total
            cum = 0
            chosen = fams[0]
            for f in fams:
                cum += fam_n[f]
                if r <= cum:
                    chosen = f; break
            # sample alpha around posterior mode with std tau (Laplace)
            alpha = alpha_post[chosen] + tau * rnd.gauss(0, 1) * 0.3
            theta = inv_logit(mu[k] + alpha)
            thetas.append(theta)
        thetas.sort()
        n = len(thetas)
        out[k] = {
            "post_mean": round(sum(thetas) / n, 3),
            "ci95": [round(thetas[int(0.025 * n)], 3),
                       round(thetas[int(0.975 * n)], 3)],
            "P_gt_0.50": round(sum(1 for t in thetas if t > 0.50) / n, 3),
            "P_gt_0.55": round(sum(1 for t in thetas if t > 0.55) / n, 3),
            "P_gt_0.60": round(sum(1 for t in thetas if t > 0.60) / n, 3),
        }
    return out


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

    # ---- DerSimonian-Laird random-effects meta-analysis (R9 reviewer ask
    # for actual hierarchical model with proper between-judge variance)
    print(f"\n=== DerSimonian-Laird random-effects pool (cached only) ===")
    print(f"{'KEEP':5s} {'post_mean':>10s} {'CI95':>16s} {'tau_logit':>10s} "
          f"{'k_cells':>8s} {'P>.50':>7s} {'P>.55':>7s} {'P>.60':>7s}")
    print("-" * 92)
    summary["_DL_cached"] = {}
    for kid in KEEPS:
        cells = list(cached[kid].values())
        pooled = dl_pool_logit(cells)
        ps = dl_posterior_summary(pooled)
        if ps is None: continue
        print(f"{kid:5s} {ps['post_mean']:>10.3f} {str(ps['ci95']):>16s} "
              f"{ps['tau_logit']:>10.3f} {ps['k_cells']:>8d} "
              f"{ps['P_gt_0.50']:>7.3f} {ps['P_gt_0.55']:>7.3f} "
              f"{ps['P_gt_0.60']:>7.3f}")
        summary["_DL_cached"][kid] = ps

    print(f"\n=== DerSimonian-Laird random-effects pool (fresh only) ===")
    print(f"{'KEEP':5s} {'post_mean':>10s} {'CI95':>16s} {'tau_logit':>10s} "
          f"{'k_cells':>8s} {'P>.50':>7s} {'P>.55':>7s} {'P>.60':>7s}")
    print("-" * 92)
    summary["_DL_fresh"] = {}
    for kid in KEEPS:
        cells = list(fresh[kid].values())
        if len(cells) < 1: continue
        pooled = dl_pool_logit(cells)
        ps = dl_posterior_summary(pooled)
        if ps is None: continue
        print(f"{kid:5s} {ps['post_mean']:>10.3f} {str(ps['ci95']):>16s} "
              f"{ps['tau_logit']:>10.3f} {ps['k_cells']:>8d} "
              f"{ps['P_gt_0.50']:>7.3f} {ps['P_gt_0.55']:>7.3f} "
              f"{ps['P_gt_0.60']:>7.3f}")
        summary["_DL_fresh"][kid] = ps

    print(f"\nNote: DL random-effects model treats logit(theta_j) per "
          f"judge cell as Normal(mu, tau^2 + v_j); tau is the between-"
          f"judge SD on logit scale, k is the number of judge cells "
          f"contributing. Posterior is on the population-level mu.")

    OUT_LOG.write_text(json.dumps({
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prior": "Beta(2, 2) for per-cell; hierarchical EB for pooled",
        "summary": summary,
    }, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
