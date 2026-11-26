"""Extract hereditary moduli and fit a scanner-aligned failure model."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def _stress_design(trace, taus):
    past_t = np.asarray(trace['past_timestamps'], dtype=float)
    past_u = np.asarray(trace['past_strain_history'], dtype=float)
    t = np.asarray(trace['timestamps'], dtype=float)
    u = np.asarray(trace['strain_u'], dtype=float)
    # This includes the interval from the last prehistory sample to t[0].
    innovation = -np.expm1(-np.diff(np.r_[past_t, t])[:, None] / taus)
    state = np.full(taus.size, past_u[0])
    split = past_t.size - 1
    for rate, strain in zip(innovation[:split], past_u[1:]):
        state += rate * (strain - state)
    memory = np.empty((t.size, taus.size))
    for k, (rate, strain) in enumerate(zip(innovation[split:], u)):
        state += rate * (strain - state)
        memory[k] = state
    return np.column_stack((np.ones(t.size), u, memory))


def _trace_moduli(trace, taus):
    design = _stress_design(trace, taus)
    stress = np.asarray(trace['stress_obs'], dtype=float)
    coef = np.linalg.lstsq(design, stress, rcond=None)[0]
    # Adaptive Cauchy weights acquire the inlier fit before hard rejection.
    for _ in range(14):
        residual = stress - design @ coef
        mad = 1.4826 * np.median(np.abs(residual - np.median(residual)))
        root_weight = 1.0 / np.hypot(1.0, residual / (2.0 * max(0.015, mad)))
        coef = np.linalg.lstsq(design * root_weight[:, None], stress * root_weight, rcond=None)[0]
    residual = stress - design @ coef
    deviation = np.abs(residual - np.median(residual))
    keep = deviation <= max(0.12, 7.0 * 1.4826 * np.median(deviation))
    refit, _, rank, _ = np.linalg.lstsq(design[keep], stress[keep], rcond=None)
    if rank == design.shape[1]:
        coef = refit
    return coef[1:]


def _aligned_moduli(partition, taus):
    moduli = np.array([_trace_moduli(trace, taus) for trace in partition['traces']])
    moduli[np.asarray(partition['missing_channel_mask'], dtype=bool)] = np.nan
    centers = np.asarray(partition['center_id'])
    controls = np.asarray(partition['controls'], dtype=float)
    for center in np.unique(centers):
        rows = centers == center
        panel = controls[rows]
        panel = panel[:, np.sum(np.isfinite(panel), axis=0) >= 2]
        means = np.nanmean(panel, axis=0)
        spreads = np.nanstd(panel, axis=0, ddof=0)
        offset = np.median(means)
        gain = np.median(spreads)
        moduli[rows] = (moduli[rows] - offset) / gain
    return moduli


def _cell_weights(center, stratum):
    _, cell, counts = np.unique(np.column_stack((center, stratum)), axis=0,
                                return_inverse=True, return_counts=True)
    weights = 1.0 / counts[cell.ravel()]
    return weights / weights.mean()


def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Return aligned query moduli, failure probabilities and biomarker slopes."""
    taus = np.asarray(cohort['taus'], dtype=float)
    train, query = cohort['train'], cohort['query']
    train_z = _aligned_moduli(train, taus)
    query_z = _aligned_moduli(query, taus)
    weights = _cell_weights(train['center_id'], train['stratum'])
    observed = np.isfinite(train_z)
    fill = (np.sum(np.where(observed, train_z, 0.0) * weights[:, None], axis=0)
            / np.sum(observed * weights[:, None], axis=0))
    train_z = np.where(observed, train_z, fill)
    query_z = np.where(np.isfinite(query_z), query_z, fill)

    # Keep the biological stratum offsets, but do not use scanner ID or balance
    # the labels themselves: the output is risk under the fitted class prior.
    strata = np.eye(3)
    train_features = np.column_stack((train_z, strata[np.asarray(train['stratum'])]))
    query_features = np.column_stack((query_z, strata[np.asarray(query['stratum'])]))
    model = LogisticRegression(C=10.0, max_iter=500, tol=1e-8)
    model.fit(train_features, train['y'], sample_weight=weights)
    probabilities = model.predict_proba(query_features)[:, 1]
    eps = np.finfo(float).eps
    return {
        'aligned_query_moduli': query_z,
        'query_probabilities': np.clip(probabilities, eps, 1.0 - eps),
        'modulus_weights': model.coef_[0, :3].copy(),
    }
