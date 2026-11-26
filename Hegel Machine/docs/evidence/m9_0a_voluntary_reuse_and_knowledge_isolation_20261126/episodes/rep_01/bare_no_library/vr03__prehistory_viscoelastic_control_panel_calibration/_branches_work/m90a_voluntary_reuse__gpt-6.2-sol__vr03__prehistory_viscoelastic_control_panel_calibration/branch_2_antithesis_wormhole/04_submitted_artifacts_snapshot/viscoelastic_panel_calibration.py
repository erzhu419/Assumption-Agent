"""Recover viscoelastic moduli and calibrate risk on the control-panel scale."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def _design(trace, taus):
    past = np.asarray(trace['past_strain_history'], dtype=float)
    strain = np.asarray(trace['strain_u'], dtype=float)
    time = np.concatenate((trace['past_timestamps'], trace['timestamps']))
    inputs = np.concatenate((past, strain))
    memory = np.empty((inputs.size, taus.size))
    memory[0] = inputs[0]
    # This includes the bridge from the last pre-window sample to the first observation.
    for k, alpha in enumerate(np.exp(-np.diff(time)[:, None] / taus), 1):
        memory[k] = alpha * memory[k - 1] + (1.0 - alpha) * inputs[k]
    return np.column_stack((np.ones(strain.size), strain, memory[past.size:]))


def _extract(block, taus, subsets):
    raw = np.empty((len(block['traces']), 3))
    for i, trace in enumerate(block['traces']):
        design = _design(trace, taus)
        stress = np.asarray(trace['stress_obs'], dtype=float)
        candidates = np.einsum('ijk,ik->ij', np.linalg.pinv(design[subsets]), stress[subsets])
        error = np.abs(stress[:, None] - design @ candidates.T)
        residual = error[:, np.argmin(np.median(error, axis=0))]
        # Identify a consensus before refitting; large glitches cannot exert leverage.
        inlier = residual <= max(0.1, 6.0 * np.median(residual))
        raw[i] = np.linalg.lstsq(design[inlier], stress[inlier], rcond=None)[0][1:]
    raw[np.asarray(block['missing_channel_mask'], dtype=bool)] = np.nan
    return raw


def _align(raw, block):
    center = np.asarray(block['center_id'])
    panels = np.asarray(block['controls'], dtype=float)
    z = np.empty_like(raw)
    for c in np.unique(center):
        rows = center == c
        panel = panels[rows]
        panel = panel[:, np.sum(np.isfinite(panel), axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0, ddof=0))
        z[rows] = (raw[rows] - offset) / gain
    return z


def _cell_index(center, stratum):
    _, index, counts = np.unique(np.column_stack((center, stratum)), axis=0,
                                 return_inverse=True, return_counts=True)
    return index, counts


def _imputation(z, cells, n_cells):
    # Average cell means, not individual specimens. An unobserved cell/channel
    # contributes no estimate for that channel.
    means = np.full((n_cells, z.shape[1]), np.nan)
    for c in range(n_cells):
        group = z[cells == c]
        counts = np.isfinite(group).sum(axis=0)
        np.divide(np.nansum(group, axis=0), counts, out=means[c], where=counts > 0)
    return np.nanmean(means, axis=0)


def _risk_design(z, stratum):
    s = np.asarray(stratum)
    return np.column_stack((z, np.ones(len(s)), s == 1, s == 2))


def _fit_risk(x, y, weights, ridge=1.0):
    penalty = np.array([ridge, ridge, ridge, 0.0, 0.0, 0.0])

    def objective(b):
        logits = x @ b
        loss = weights @ np.logaddexp(0, (1 - 2 * y) * logits)
        return loss + 0.5 * np.dot(penalty * b, b), x.T @ (weights * (expit(logits) - y)) + penalty * b

    return minimize(objective, np.zeros(x.shape[1]), jac=True, method='BFGS').x


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Estimate query moduli and failure probabilities without query labels."""
    taus = np.asarray(cohort['taus'], dtype=float)
    train, query = cohort['train'], cohort['query']
    rng = np.random.default_rng(518)
    subsets = np.array([rng.choice(36, 4, replace=False) for _ in range(96)])
    z_train = _align(_extract(train, taus, subsets), train)
    z_query = _align(_extract(query, taus, subsets), query)
    cells, counts = _cell_index(train['center_id'], train['stratum'])
    means = _imputation(z_train, cells, len(counts))
    z_train = np.where(np.isnan(z_train), means, z_train)
    z_query = np.where(np.isnan(z_query), means, z_query)
    weights = (len(z_train) / len(counts)) / counts[cells]
    fitted = _fit_risk(_risk_design(z_train, train['stratum']), np.asarray(train['y']), weights)
    probabilities = expit(_risk_design(z_query, query['stratum']) @ fitted)
    eps = np.finfo(float).eps
    return {'aligned_query_moduli': z_query,
            'query_probabilities': np.clip(probabilities, eps, 1.0 - eps),
            'modulus_weights': fitted[:3]}
