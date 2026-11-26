"""Hereditary modulus extraction and reference-panel risk calibration."""

def __hegel_load_skill_autogenous_prim_reference_panel_balancing():
    import types as __hegel_types
    import numpy as np
    import numpy as np

    def _real_array(x: 'np.ndarray', name: 'str') -> 'np.ndarray':
        x = np.asarray(x)
        if x.dtype.kind not in 'biuf':
            raise ValueError(f'{name} must be a real numeric array')
        return x

    def align_reference_panel(values: 'np.ndarray', controls: 'np.ndarray', groups: 'np.ndarray') -> 'np.ndarray':
        """Return aligned real values (n, p) using real controls (n, q) and finite real groups (n,), with nonempty columns and a supported positive control gain per group."""
        values = _real_array(values, 'values').astype(float, copy=False)
        controls = _real_array(controls, 'controls').astype(float, copy=False)
        groups = _real_array(groups, 'groups')
        if values.ndim != 2 or controls.ndim != 2 or (not values.shape[1]) or (not controls.shape[1]):
            raise ValueError('values and controls must be matrices with nonempty columns')
        n = values.shape[0]
        if controls.shape[0] != n or groups.shape != (n,) or (not np.isfinite(groups).all()):
            raise ValueError('controls and finite one-dimensional groups must match value rows')
        (levels, codes) = np.unique(groups, return_inverse=True)
        result = np.full(values.shape, np.nan)
        for (code, level) in enumerate(levels):
            rows = codes == code
            panel = controls[rows]
            observed = np.isfinite(panel)
            counts = observed.sum(axis=0)
            keep = counts >= 2
            if not keep.any():
                raise ValueError(f'group {level!r} has no control with two finite observations')
            (panel, observed, counts) = (panel[:, keep], observed[:, keep], counts[keep])
            means = np.where(observed, panel, 0.0).sum(axis=0) / counts
            residuals = np.where(observed, panel, means) - means
            spreads = np.sqrt(np.sum(residuals ** 2, axis=0) / counts)
            (offset, gain) = (np.median(means), np.median(spreads))
            if not np.isfinite(offset) or not np.isfinite(gain) or gain <= 0:
                raise ValueError(f'group {level!r} requires a finite offset and positive finite gain')
            block = values[rows]
            result[rows] = (np.where(np.isfinite(block), block, np.nan) - offset) / gain
        return result

    def balanced_cell_weights(groups: 'np.ndarray', strata: 'np.ndarray') -> 'np.ndarray':
        """Return positive mean-one weights (n,) from nonempty finite real vectors groups (n,) and strata (n,) with every observed group–stratum combination present."""
        groups = _real_array(groups, 'groups')
        strata = _real_array(strata, 'strata')
        if groups.ndim != 1 or not groups.size or strata.shape != groups.shape:
            raise ValueError('groups and strata must be matching nonempty vectors')
        if not np.isfinite(groups).all() or not np.isfinite(strata).all():
            raise ValueError('groups and strata must be finite')
        (group_levels, group_codes) = np.unique(groups, return_inverse=True)
        (stratum_levels, stratum_codes) = np.unique(strata, return_inverse=True)
        pairs = np.column_stack((group_codes, stratum_codes))
        (_, inverse, counts) = np.unique(pairs, axis=0, return_inverse=True, return_counts=True)
        if counts.size != group_levels.size * stratum_levels.size:
            raise ValueError('every group must contain every observed stratum')
        return groups.size / counts.size / counts[inverse.reshape(-1)]

    def weighted_column_fill(values: 'np.ndarray', weights: 'np.ndarray') -> 'np.ndarray':
        """Return finite-entry weighted means (p,) for a nonempty real matrix values (n, p) with finite positive weights (n,), using zero for unobserved columns."""
        values = _real_array(values, 'values').astype(float, copy=False)
        weights = _real_array(weights, 'weights').astype(float, copy=False)
        if values.ndim != 2 or not values.shape[0] or (not values.shape[1]):
            raise ValueError('values must be a nonempty matrix')
        if weights.shape != (values.shape[0],):
            raise ValueError('weights must be a vector matching value rows')
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError('weights must be finite and strictly positive')
        observed = np.isfinite(values)
        support = weights @ observed
        numerator = weights @ np.where(observed, values, 0.0)
        if not np.isfinite(support).all() or not np.isfinite(numerator).all():
            raise ValueError('weighted column sums must be finite')
        return np.divide(numerator, support, out=np.zeros(values.shape[1]), where=support > 0)
    return __hegel_types.SimpleNamespace(align_reference_panel=align_reference_panel, balanced_cell_weights=balanced_cell_weights, weighted_column_fill=weighted_column_fill)
__hegel_skill_ns_autogenous_prim_reference_panel_balancing = __hegel_load_skill_autogenous_prim_reference_panel_balancing()
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
align_reference_panel = __hegel_skill_ns_autogenous_prim_reference_panel_balancing.align_reference_panel
balanced_cell_weights = __hegel_skill_ns_autogenous_prim_reference_panel_balancing.balanced_cell_weights
weighted_column_fill = __hegel_skill_ns_autogenous_prim_reference_panel_balancing.weighted_column_fill

def _stress_design(trace, taus):
    t = np.r_[trace['past_timestamps'], trace['timestamps']]
    u = np.r_[trace['past_strain_history'], trace['strain_u']]
    tau = np.asarray(taus, dtype=float)
    dt = np.diff(t)[:, None] / tau
    gain = -np.expm1(-dt)
    memory = np.empty((len(t), len(tau)))
    memory[0] = u[0]
    for (k, g) in enumerate(gain, 1):
        memory[k] = memory[k - 1] + g * (u[k] - memory[k - 1])
    n = len(trace['timestamps'])
    return np.column_stack((np.ones(n), u[-n:], memory[-n:]))

def _robust_moduli(design, stress, subsets):
    (q, r) = np.linalg.qr(design, mode='reduced')
    candidates = np.linalg.pinv(q[subsets]) @ stress[subsets, None]
    candidates = np.column_stack((q.T @ stress, candidates[..., 0].T))
    errors = (stress[:, None] - q @ candidates) ** 2
    h = int(np.ceil(0.8 * len(stress)))
    scores = np.partition(errors, h - 1, axis=0)[:h].sum(axis=0)
    theta = candidates[:, scores.argmin()]
    for _ in range(4):
        keep = np.argpartition(np.abs(stress - q @ theta), h - 1)[:h]
        theta = np.linalg.lstsq(q[keep], stress[keep], rcond=None)[0]
    residual = stress - q @ theta
    scale = max(1e-09, 1.4826 * np.median(np.abs(residual - np.median(residual))))
    for _ in range(12):
        w = 1.0 / np.hypot(1.0, residual / (2.5 * scale))
        new = np.linalg.lstsq(q * w[:, None], stress * w, rcond=None)[0]
        if np.linalg.norm(new - theta) <= 1e-10 * (1 + np.linalg.norm(theta)):
            theta = new
            break
        theta = new
        residual = stress - q @ theta
    return np.linalg.lstsq(r, theta, rcond=None)[0][1:]

def _aligned(split, taus):
    rng = np.random.default_rng(532)
    n = len(split['traces'][0]['timestamps'])
    subsets = np.array([rng.choice(n, 7, replace=False) for _ in range(48)])
    raw = np.array([_robust_moduli(_stress_design(trace, taus), np.asarray(trace['stress_obs'], dtype=float), subsets) for trace in split['traces']])
    raw[np.asarray(split['missing_channel_mask'], dtype=bool)] = np.nan
    return align_reference_panel(raw, split['controls'], split['center_id'])

def _features(z, s):
    return np.column_stack((z, np.asarray(s)[:, None] == np.arange(1, 3)))

def calibrate_viscoelastic_cohort(cohort: dict) -> dict:
    """Recover query moduli and estimate stratum-adjusted failure probabilities."""
    (train, query) = (cohort['train'], cohort['query'])
    z = _aligned(train, cohort['taus'])
    zq = _aligned(query, cohort['taus'])
    w = balanced_cell_weights(train['center_id'], train['stratum'])
    complete_train = np.isfinite(z).all(axis=1)
    missing_query = np.isnan(zq)
    fill = weighted_column_fill(z, w)
    z = np.where(np.isnan(z), fill, z)
    zq = np.where(np.isnan(zq), fill, zq)
    model = LogisticRegression(C=5.0, max_iter=500, tol=1e-09)
    model.fit(_features(z, train['stratum']), train['y'], sample_weight=w)
    logit = model.decision_function(_features(zq, query['stratum']))
    prob = expit(logit)
    incomplete = missing_query.any(axis=1)
    if incomplete.any() and complete_train.sum() >= 2:
        # Mean imputation fixes the logit center, not its uncertainty.
        cov = np.cov(z[complete_train], rowvar=False,
                     aweights=w[complete_train], ddof=0)
        beta_missing = missing_query[incomplete] * model.coef_[0, :3]
        variance = np.einsum('ij,jk,ik->i', beta_missing, cov, beta_missing)
        nodes, weights = np.polynomial.hermite.hermgauss(24)
        offsets = np.sqrt(2 * np.maximum(0, variance))[:, None] * nodes
        prob[incomplete] = expit(logit[incomplete, None] + offsets) @ (weights / np.sqrt(np.pi))
    return {'aligned_query_moduli': zq, 'query_probabilities': np.clip(prob, 1e-12, 1.0 - 1e-12), 'modulus_weights': model.coef_[0, :3].copy()}
