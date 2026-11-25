"""Reference-control alignment and stratum-aware multicenter assay prediction."""

import numpy as np
from scipy.special import digamma, polygamma
from sklearn.linear_model import LogisticRegression


def _excess_dispersion(estimates, noise):
    """Separate between-biomarker dispersion from control sampling noise."""
    precision = 1.0 / noise
    total = precision.sum()
    mean = np.dot(precision, estimates) / total
    q = np.dot(precision, (estimates - mean) ** 2)
    dof = len(estimates) - 1
    information = max(0.0, total - np.dot(precision, precision) / total)
    return q - dof, information


def _shrink(estimates, noise):
    """Shrink noisy biomarker estimates toward the local reference panel."""
    dispersion = [_excess_dispersion(x, v) for x, v in zip(estimates, noise)
                  if len(x) > 1]
    numerator, denominator = np.sum(dispersion, axis=0) if dispersion else (0.0, 0.0)
    shared = max(0.0, numerator / denominator) if denominator else 0.0
    fitted = []
    for x, v in zip(estimates, noise):
        if not len(x):
            fitted.append(x)
            continue
        excess, information = _excess_dispersion(x, v)
        local = max(0.0, excess / information) if information else 0.0
        # Strong local heterogeneity can justify less pooling than the full panel.
        spread = max(shared, local)
        center = np.median(x)
        if spread > 0:
            fitted.append(center + spread / (spread + v) * (x - center))
        else:
            fitted.append(np.full_like(x, center))
    return fitted


def _reference_affines(reference):
    raw = np.asarray(reference['X'], dtype=float)
    group = np.asarray(reference['center_id'])
    centers = np.unique(group)
    n = np.zeros((len(centers), raw.shape[1]), dtype=int)
    loc = np.zeros_like(n, dtype=float)
    log_scale = np.zeros_like(loc)
    usable = np.zeros_like(n, dtype=bool)
    for i, c in enumerate(centers):
        for j, column in enumerate(raw[group == c].T):
            observed = column[np.isfinite(column)]
            n[i, j] = len(observed)
            if len(observed) >= 2:
                sd = observed.std(ddof=1)
                if sd > 0:
                    df = len(observed) - 1
                    loc[i, j] = observed.mean()
                    log_scale[i, j] = np.log(sd) - (digamma(df / 2) - np.log(df / 2)) / 2
                    usable[i, j] = True

    scale_fits = _shrink([a[m] for a, m in zip(log_scale, usable)],
                         [polygamma(1, (a[m] - 1) / 2) / 4 for a, m in zip(n, usable)])
    scale = np.ones_like(loc)
    normalized_loc, loc_noise, center_scales = [], [], []
    for i, m in enumerate(usable):
        s = np.exp(scale_fits[i])
        typical = np.median(s) if len(s) else 1.0
        scale[i] = typical
        scale[i, m] = s
        center_scales.append(typical)
        normalized_loc.append(loc[i, m] / typical)
        loc_noise.append((s / typical) ** 2 / n[i, m])
    loc_fits = _shrink(normalized_loc, loc_noise)
    fitted_loc = np.zeros_like(loc)
    for i, m in enumerate(usable):
        if m.any():
            fitted = loc_fits[i] * center_scales[i]
            fitted_loc[i] = np.median(fitted)
            fitted_loc[i, m] = fitted

    pooled_loc = np.zeros(raw.shape[1])
    pooled_scale = np.ones(raw.shape[1])
    for j, m in enumerate(usable.T):
        if m.any():
            pooled_loc[j] = np.median(fitted_loc[m, j])
            pooled_scale[j] = np.median(scale[m, j])
        elif usable.any():
            pooled_loc[j] = np.median(fitted_loc[usable])
            pooled_scale[j] = np.median(scale[usable])
    affines = {c: (fitted_loc[i], scale[i]) for i, c in enumerate(centers) if usable[i].any()}
    return affines, (pooled_loc, pooled_scale)


def _align(part, affines, pooled):
    raw = np.asarray(part['X'], dtype=float)
    groups = np.asarray(part['center_id'])
    z = np.empty_like(raw)
    for c in np.unique(groups):
        rows = groups == c
        loc, scale = affines.get(c, pooled)
        z[rows] = (raw[rows] - loc) / scale
    return np.where(np.isfinite(z), z, 0.0)


def calibrate_multicenter_assay(dataset: dict) -> dict:
    """Align with unlabeled reference controls and fit patient probabilities."""
    train, query = dataset['train'], dataset['query']
    affines, pooled = _reference_affines(dataset['reference_controls'])
    x = _align(train, affines, pooled)
    z = _align(query, affines, pooled)
    levels = np.unique(train['stratum'])

    def design(features, part):
        # Relax the penalty on a small set of clinical stratum intercepts.
        strata = 3.0 * (np.asarray(part['stratum'])[:, None] == levels)
        return np.column_stack((features, strata))

    model = LogisticRegression(C=0.25, max_iter=500, tol=1e-8)
    model.fit(design(x, train), train['y'])
    probability = model.predict_proba(design(z, query))[:, 1] if len(z) else np.empty(0)
    return {
        'aligned_query_features': z,
        'query_probabilities': np.clip(probability, 1e-12, 1.0 - 1e-12),
        'feature_weights': model.coef_[0, :x.shape[1]].copy(),
    }
