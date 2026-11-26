"""Prehistory-aware viscoelastic calibration and control-referenced risk."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _stress_design(trace, taus):
    t = np.asarray(trace['timestamps'], dtype=float)
    past_t = np.asarray(trace['past_timestamps'], dtype=float)
    past_u = np.asarray(trace['past_strain_history'], dtype=float)
    u = np.asarray(trace['strain_u'], dtype=float)
    times = np.r_[past_t, t]
    history = np.r_[past_u, u]
    step = -np.expm1(-np.diff(times)[:, None] / taus)
    memory = np.full(taus.shape, history[0])
    design = np.empty((t.size, 4))
    design[:, 0] = 1.0
    design[:, 1] = u
    for k, alpha in enumerate(step, start=1):
        memory += alpha * (history[k] - memory)
        if k >= past_t.size:
            design[k - past_t.size, 2:] = memory
    return design


def _robust_moduli(a, y, subsets):
    # A trimmed seed keeps large glitches from setting the Cauchy scale.
    best = np.linalg.lstsq(a, y, rcond=None)[0]
    keep = (2 * y.size) // 3

    def score(b):
        r2 = (y - a @ b) ** 2
        return np.partition(r2, keep - 1)[:keep].sum()

    loss = score(best)
    for idx in subsets:
        b, _, rank, _ = np.linalg.lstsq(a[idx], y[idx], rcond=None)
        if rank == a.shape[1]:
            val = score(b)
            if val < loss:
                best, loss = b, val
    r = y - a @ best
    scale = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-8)
    for _ in range(24):
        w = 1.0 / np.hypot(1.0, (y - a @ best) / scale)
        b = np.linalg.lstsq(a * w[:, None], y * w, rcond=None)[0]
        if np.max(np.abs(a @ (b - best))) <= 1e-6 * scale:
            best = b
            break
        best = b
    # Refit the clean samples when residuals clearly separate from telemetry spikes.
    r = y - a @ best
    clean = np.abs(r - np.median(r)) <= 6.0 * scale
    b, _, rank, _ = np.linalg.lstsq(a[clean], y[clean], rcond=None)
    return (b if rank == a.shape[1] else best)[1:]


def _aligned_moduli(split, taus):
    traces = split['traces']
    n = len(traces[0]['timestamps'])
    rng = np.random.default_rng(321)
    subsets = [rng.choice(n, 8, replace=False) for _ in range(36)]
    x = np.array([_robust_moduli(_stress_design(trace, taus),
                               np.asarray(trace['stress_obs'], dtype=float), subsets)
                  for trace in traces])
    x[np.asarray(split['missing_channel_mask'], dtype=bool)] = np.nan
    center = np.asarray(split['center_id'])
    controls = np.asarray(split['controls'], dtype=float)
    for c in np.unique(center):
        rows = center == c
        panel = controls[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        spread = np.nanstd(panel, axis=0)
        valid = spread > 0
        if not np.any(valid):
            raise ValueError(f'Center {c} has no usable reference control')
        location = np.nanmean(panel[:, valid], axis=0)
        x[rows] = (x[rows] - np.median(location)) / np.median(spread[valid])
    return x


def _cell_weights(center, strata):
    _, groups, counts = np.unique(np.column_stack((center, strata)), axis=0,
                                  return_inverse=True, return_counts=True)
    w = 1.0 / counts[groups]
    return w / w.mean()


def _risk(x, y, center, strata, query, query_strata):
    w = _cell_weights(center, strata)
    finite = np.isfinite(x)
    denominator = w @ finite
    fill = np.divide(w @ np.where(finite, x, 0.0), denominator,
                     out=np.zeros(x.shape[1]), where=denominator > 0)
    x = np.where(finite, x, fill)
    query = np.where(np.isfinite(query), query, fill)
    levels = np.unique(strata)

    def design(z, s):
        return np.column_stack((np.ones(len(s)), z, s[:, None] == levels[1:]))

    a = design(x, strata)
    aq = design(query, query_strata)
    penalty = np.zeros(a.shape[1])
    penalty[1:4] = 1.0 / 3.0

    def objective(b):
        z = a @ b
        loss = w @ (np.logaddexp(0, z) - y * z) + 0.5 * (penalty * b) @ b
        grad = a.T @ (w * (expit(z) - y)) + penalty * b
        return loss, grad

    fit = minimize(objective, np.zeros(a.shape[1]), jac=True, method='L-BFGS-B',
                   options={'ftol': 1e-13, 'gtol': 1e-8})
    prob = np.clip(expit(aq @ fit.x), 1e-15, 1 - 1e-15)
    return query, prob, fit.x[1:4]


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Return imputed query moduli, failure probabilities, and log-odds weights."""
    taus = np.asarray(cohort['taus'], dtype=float)
    train, query = cohort['train'], cohort['query']
    x = _aligned_moduli(train, taus)
    z = _aligned_moduli(query, taus)
    z, prob, beta = _risk(x, np.asarray(train['y']), np.asarray(train['center_id']),
                          np.asarray(train['stratum']), z, np.asarray(query['stratum']))
    return {'aligned_query_moduli': z, 'query_probabilities': prob, 'modulus_weights': beta}
