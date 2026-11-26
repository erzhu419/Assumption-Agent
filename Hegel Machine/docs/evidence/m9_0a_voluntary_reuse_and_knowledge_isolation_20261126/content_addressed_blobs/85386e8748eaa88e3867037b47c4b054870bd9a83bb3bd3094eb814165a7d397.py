"""Hereditary stress extraction and control-panel risk calibration."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(trace, taus):
    t = np.r_[trace['past_timestamps'], trace['timestamps']]
    u = np.r_[trace['past_strain_history'], trace['strain_u']]
    past = len(trace['past_timestamps'])
    alpha = -np.expm1(-np.diff(t)[:, None] / taus)
    state = np.full(len(taus), u[0])
    x = np.empty((len(t) - past, 1 + len(taus)))
    for k, a in enumerate(alpha, start=1):
        state += a * (u[k] - state)
        if k >= past:
            x[k - past, 0] = u[k]
            x[k - past, 1:] = state
    return np.column_stack((np.ones(len(x)), x))


def _robust_fit(a, y, subsets):
    # Whiten the design for stable minimal-subset fits, even when the
    # different Prony modes have very different observed amplitudes.
    q, r = np.linalg.qr(a, mode='reduced')
    sub = q[subsets]
    good = np.abs(np.linalg.det(sub)) > 1e-10
    idx = subsets[good]
    starts = np.linalg.solve(sub[good], y[idx][..., None])[..., 0]
    starts = np.vstack((starts, q.T @ y))
    trimmed = int(0.72 * len(y))
    resid = y[:, None] - q @ starts.T
    error = np.partition(resid**2, trimmed - 1, axis=0)[:trimmed].sum(axis=0)
    b = starts[error.argmin()]
    for _ in range(5):
        keep = np.argsort(np.abs(y - q @ b))[:trimmed]
        b = np.linalg.lstsq(q[keep], y[keep], rcond=None)[0]
    for _ in range(4):
        resid = y - q @ b
        med = np.median(resid)
        scale = 1.4826 * np.median(np.abs(resid - med))
        keep = np.abs(resid - med) <= max(4.5 * scale, 1e-10)
        b = np.linalg.lstsq(q[keep], y[keep], rcond=None)[0]
    return np.linalg.solve(r, b)[1:]


def _aligned(split, taus):
    center = np.asarray(split['center_id'])
    controls = np.asarray(split['controls'], dtype=float)
    rng = np.random.default_rng(185)
    subsets = np.array([rng.choice(36, 4, replace=False) for _ in range(96)])
    x = np.array([_robust_fit(_design(tr, taus), np.asarray(tr['stress_obs']), subsets)
                  for tr in split['traces']])
    for c in np.unique(center):
        rows = center == c
        panel = controls[rows]
        loc = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0))
        x[rows] = (x[rows] - loc) / scale
    x[np.asarray(split['missing_channel_mask'], dtype=bool)] = np.nan
    return x


def _cell_stats(x, center, stratum):
    means, groups = [], []
    for c in np.unique(center):
        for s in np.unique(stratum[center == c]):
            group = np.flatnonzero((center == c) & (stratum == s))
            means.append(np.nanmean(x[group], axis=0))
            groups.append(group)
    weights = np.empty(len(x))
    for group in groups:
        weights[group] = len(x) / (len(groups) * len(group))
    return np.nanmean(means, axis=0), weights


def _logistic(x, y, weight):
    def objective(b):
        eta = x @ b
        loss = weight @ (np.logaddexp(0, eta) - y * eta)
        grad = x.T @ (weight * (expit(eta) - y))
        return loss, grad
    return minimize(objective, np.zeros(x.shape[1]), jac=True, method='BFGS',
                    options={'gtol': 1e-7}).x


def _predict(x, xq, train, query):
    center = np.asarray(train['center_id'])
    strata = np.asarray(train['stratum'])
    means, weight = _cell_stats(x, center, strata)
    x = np.where(np.isnan(x), means, x)
    xq = np.where(np.isnan(xq), means, xq)
    b = _logistic(np.column_stack((x, np.eye(3)[strata])), np.asarray(train['y']), weight)
    pred = expit(xq @ b[:3] + b[3:][np.asarray(query['stratum'])])
    return xq, np.clip(pred, 1e-15, 1 - 1e-15), b[:3]


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Align raw moduli and predict failure without using query outcomes."""
    taus = np.asarray(cohort['taus'], dtype=float)
    train, query = cohort['train'], cohort['query']
    x, xq = _aligned(train, taus), _aligned(query, taus)
    moduli, probability, weights = _predict(x, xq, train, query)
    return {'aligned_query_moduli': moduli, 'query_probabilities': probability,
            'modulus_weights': weights}
