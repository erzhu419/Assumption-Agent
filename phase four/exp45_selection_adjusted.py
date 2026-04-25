"""Exp 45 — Selection-adjusted (winner's curse) analysis on the
inner-loop top-3 of 12 candidates.

The 3 KEEPs (W076=0.64, W077=0.60, W078=0.60) are the maxima of
12 noisy n=50 measurements. Under any reasonable null where true
candidate wr's are clustered around 0.50, the OBSERVED maxima are
inflated; expected drop on remeasurement (regression to the mean)
exists even with no judge-fragility.

We do two complementary analyses:

(1) Empirical-Bayes shrinkage: fit a Beta(alpha, beta) prior to
    the 12 observed wr's via maximum marginal likelihood, then
    compute the posterior expected wr for each candidate. The
    delta {observed - posterior_mean} for the top 3 is the
    expected regression-to-the-mean drop attributable to
    selection alone.

(2) Monte Carlo: simulate 10,000 draws of 12 candidates with
    true wr's ~ Beta(prior), simulate n=50 observed wr per
    candidate, sort, and report the expected maximum, second,
    and third observed values vs their true wr. Compare the
    observed inner-loop -> L1 drops (W076: 0.24, W077: 0.13,
    W078: 0.09) to the simulated regression-to-mean band.

If the simulated drops are similar to or larger than what we
observed for L1, the entire L1 drop can be explained by
selection bias alone (no judge-fragility needed). If observed
drops exceed the simulated band, judge-fragility contributes
beyond pure selection.
"""

import json
import math
from pathlib import Path
import numpy as np
from scipy import stats, optimize

PROJECT = Path(__file__).parent.parent
AUTO = PROJECT / "phase four" / "autonomous"
OUT_LOG = AUTO / "exp45_selection_adjusted_log.json"


def load_inner_loop_wrs():
    """Return {tid: (k, n)} from the validation log; tid -> (ext wins, total)."""
    out = {}
    val = json.loads((AUTO / "validation_log_parallel.json").read_text())
    for entry in val:
        for r in entry["results"]:
            ab = r["ab"]
            ne, nb, nt = ab["wins_a"], ab["wins_b"], ab.get("ties", 0)
            out[r["tid"]] = (ne, ne + nb)
    return out


def fit_beta_via_marginal_likelihood(observed_pairs):
    """Empirical-Bayes: fit Beta(alpha, beta) prior to wr's via MLE on
    the marginal likelihood of (k_i, n_i) under Beta-Binomial.
    Returns (alpha, beta)."""
    from scipy.special import betaln, gammaln

    def neg_log_marginal(params):
        a, b = math.exp(params[0]), math.exp(params[1])  # positivity
        ll = 0.0
        for k, n in observed_pairs:
            # Beta-Binomial PMF
            ll += (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
                    + betaln(k + a, n - k + b) - betaln(a, b))
        return -ll

    res = optimize.minimize(neg_log_marginal, x0=[0.0, 0.0], method="Nelder-Mead")
    a, b = math.exp(res.x[0]), math.exp(res.x[1])
    return a, b


def posterior_mean(k, n, a, b):
    """Posterior mean for Beta(a + k, b + n - k)."""
    return (a + k) / (a + b + n)


def main():
    obs = load_inner_loop_wrs()
    print(f"Inner-loop observations: {len(obs)} candidates")
    obs_pairs = [(k, n) for k, n in obs.values()]
    print(f"Observed (k, n) pairs: {obs_pairs}\n")

    # --- (1) Empirical-Bayes ---
    a, b = fit_beta_via_marginal_likelihood(obs_pairs)
    prior_mean = a / (a + b)
    print(f"=== Empirical-Bayes Beta prior ===")
    print(f"  fit Beta(alpha={a:.3f}, beta={b:.3f})")
    print(f"  prior mean = {prior_mean:.3f}")
    print(f"  prior concentration (alpha + beta) = {a+b:.3f}")
    print(f"  (low concentration => high shrinkage; high => low shrinkage)")

    print(f"\n=== Per-candidate observed vs shrunk posterior mean ===")
    print(f"{'tid':12s} {'observed':>10s} {'shrunk':>10s} {'shrinkage':>10s}")
    sorted_obs = sorted(obs.items(), key=lambda kv: -(kv[1][0] / kv[1][1]))
    for tid, (k, n) in sorted_obs:
        wr_obs = k / n
        wr_post = posterior_mean(k, n, a, b)
        shrink = wr_obs - wr_post
        marker = "  <-- KEEP" if tid in ("WCAND05", "WCAND10", "WCROSSL01") else ""
        print(f"{tid:12s} {wr_obs:>10.3f} {wr_post:>10.3f} {shrink:>+10.3f}{marker}")

    # --- (2) Monte Carlo simulation ---
    print(f"\n=== Monte Carlo: top-3-of-12 selection effect ===")
    rng = np.random.RandomState(42)
    n_sim = 10000
    K = 12   # candidates
    n_obs = 50  # n per candidate
    top_k = 3

    expected_obs_top = np.zeros((n_sim, top_k))
    expected_true_top = np.zeros((n_sim, top_k))
    expected_remeasure = np.zeros((n_sim, top_k))

    for s in range(n_sim):
        # Draw 12 true wr's from the fitted Beta
        true_wrs = rng.beta(a, b, size=K)
        # Observe each via Binomial(n_obs, true)
        obs_counts = rng.binomial(n_obs, true_wrs)
        obs_wrs = obs_counts / n_obs
        # Sort by OBSERVED wr (selection step)
        order = np.argsort(-obs_wrs)
        for i in range(top_k):
            j = order[i]
            expected_obs_top[s, i] = obs_wrs[j]
            expected_true_top[s, i] = true_wrs[j]
            # Remeasurement at the same true wr, fresh n_obs sample
            remeasure = rng.binomial(n_obs, true_wrs[j]) / n_obs
            expected_remeasure[s, i] = remeasure

    print(f"  (n_sim={n_sim}, K={K} candidates, n_obs={n_obs}, top_k={top_k})")
    print(f"  fitted Beta({a:.2f}, {b:.2f})")
    print(f"\n  Expected values across simulations:")
    print(f"    top-1: obs mean={expected_obs_top[:,0].mean():.3f}, "
          f"true={expected_true_top[:,0].mean():.3f}, "
          f"remeasure={expected_remeasure[:,0].mean():.3f}")
    print(f"    top-2: obs mean={expected_obs_top[:,1].mean():.3f}, "
          f"true={expected_true_top[:,1].mean():.3f}, "
          f"remeasure={expected_remeasure[:,1].mean():.3f}")
    print(f"    top-3: obs mean={expected_obs_top[:,2].mean():.3f}, "
          f"true={expected_true_top[:,2].mean():.3f}, "
          f"remeasure={expected_remeasure[:,2].mean():.3f}")

    # Expected drops on remeasurement
    print(f"\n  Expected drop on remeasurement (selection-bias only, no judge-fragility):")
    drops = expected_obs_top - expected_remeasure
    for i in range(top_k):
        d = drops[:, i]
        print(f"    top-{i+1}: drop mean={d.mean():+.3f}, "
              f"95% range [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]")

    # Compare to observed L1 drops (cross-family Opus)
    obs_drops = {"W076": 0.64 - 0.40, "W077": 0.60 - 0.475, "W078": 0.60 - 0.511}
    print(f"\n  Observed inner-loop -> L1 drops (cross-family Opus):")
    print(f"    W076: {obs_drops['W076']:+.3f}")
    print(f"    W077: {obs_drops['W077']:+.3f}")
    print(f"    W078: {obs_drops['W078']:+.3f}")

    sim_drops_central = drops.mean(axis=0)
    sim_drops_p975 = np.percentile(drops, 97.5, axis=0)
    print(f"\n  Selection-only expected drop:")
    print(f"    top-1 mean = {sim_drops_central[0]:+.3f} (95th pct = {sim_drops_p975[0]:+.3f})")
    print(f"    top-2 mean = {sim_drops_central[1]:+.3f} (95th pct = {sim_drops_p975[1]:+.3f})")
    print(f"    top-3 mean = {sim_drops_central[2]:+.3f} (95th pct = {sim_drops_p975[2]:+.3f})")

    # Headline interpretation
    print(f"\n=== Headline interpretation ===")
    for i, (kp, drop) in enumerate(obs_drops.items()):
        sim_p975 = sim_drops_p975[i]
        sim_mean = sim_drops_central[i]
        if drop <= sim_mean:
            v = "fully explained by selection bias alone"
        elif drop <= sim_p975:
            v = "within selection-bias 95% range"
        else:
            v = "EXCEEDS selection-bias 97.5%-tile -> judge-fragility component beyond pure selection"
        print(f"  {kp} drop {drop:+.3f}: {v}")

    out = {
        "timestamp": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fitted_beta": {"alpha": float(a), "beta": float(b),
                          "prior_mean": float(prior_mean)},
        "per_candidate": {tid: {"k": k, "n": n,
                                  "wr_observed": k / n,
                                  "wr_posterior_mean": posterior_mean(k, n, a, b),
                                  "shrinkage": k/n - posterior_mean(k, n, a, b)}
                            for tid, (k, n) in obs.items()},
        "monte_carlo": {
            "n_sim": n_sim,
            "expected_obs_top": [float(expected_obs_top[:, i].mean()) for i in range(top_k)],
            "expected_true_top": [float(expected_true_top[:, i].mean()) for i in range(top_k)],
            "expected_remeasure_top": [float(expected_remeasure[:, i].mean()) for i in range(top_k)],
            "expected_drop_top": [float(drops[:, i].mean()) for i in range(top_k)],
            "expected_drop_p975_top": [float(sim_drops_p975[i]) for i in range(top_k)],
        },
        "observed_drops": obs_drops,
    }
    OUT_LOG.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {OUT_LOG.name}")


if __name__ == "__main__":
    main()
