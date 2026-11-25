"""Reference-panel alignment and stratum-aware multicenter assay calibration."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, expit, polygamma


def _shrink(values, variance):
    """Estimate a common panel anchor and shrink noisy column deviations."""
    if values.size == 1:
        return values.copy(), values[0]
    precision = 1. / variance
    fixed_anchor = np.average(values, weights=precision)
    dispersion = precision @ (values - fixed_anchor)**2
    total = precision.sum()
    denominator = total - (precision @ precision) / total
    between = max(.25 * values.size / total,
                  (dispersion - (values.size - 1)) / denominator)
    anchor = np.average(values, weights=1. / (variance + between))
    return anchor + between / (between + variance) * (values - anchor), anchor


def _reference_transforms(reference):
    centers = np.asarray(reference['center_id'])
    matrix = np.asarray(reference['X'], dtype=float)
    panels = {}
    for center in np.unique(centers):
        block = matrix[centers == center]
        valid, means, scales, counts = [], [], [], []
        for j, column in enumerate(block.T):
            column = column[np.isfinite(column)]
            if column.size >= 2:
                sd = column.std(ddof=1)
                if sd > 0:
                    valid.append(j)
                    means.append(column.mean())
                    scales.append(sd)
                    counts.append(column.size)
        if not valid:
            continue
        m, sd, n = map(np.asarray, (means, scales, counts))
        # The reference model is Gaussian, so (n-1)*sample_variance / true_variance
        # is chi-square. Its log moments correct the scale bias, including at n=2.
        shape = .5 * (n - 1.)
        log_sd = np.log(sd) + .5 * (np.log(shape) - digamma(shape))
        l_fit, l_anchor = _shrink(log_sd, .25 * polygamma(1, shape))
        # Use the pooled gain for location uncertainty rather than trusting a
        # small column's observed spread to determine its own precision.
        m_fit, m_anchor = _shrink(m, np.exp(2 * l_anchor) / n)
        location = np.full(matrix.shape[1], m_anchor)
        scale = np.full(matrix.shape[1], np.exp(l_anchor))
        location[valid], scale[valid] = m_fit, np.exp(l_fit)
        panels[center] = location, scale
    if not panels:
        raise ValueError('No usable reference controls')
    locations, scales = zip(*panels.values())
    return panels, (np.median(locations, axis=0), np.median(scales, axis=0))


def _align(patient, panels, pooled):
    centers = np.asarray(patient['center_id'])
    raw = np.asarray(patient['X'], dtype=float)
    aligned = np.empty_like(raw)
    for center in np.unique(centers):
        take = centers == center
        location, scale = panels.get(center, pooled)
        aligned[take] = (raw[take] - location) / scale
    aligned[np.isnan(raw)] = 0.0
    return aligned


def _design(z, stratum):
    s = np.asarray(stratum)
    return np.column_stack((np.ones(len(s)), z, s == 1, s == 2))


def _fit(z, stratum, centers, y):
    base = _design(z, stratum)
    _, membership = np.unique(centers, return_inverse=True)
    effects = membership[:, None] == np.unique(membership)
    design = np.column_stack((base, effects))
    # Center effects are nuisance terms: estimate their dispersion from the
    # training data, but never add them to risk predictions for new centers.
    n_centers = effects.shape[1]
    k = base.shape[1]
    penalty = np.r_[0., np.full(z.shape[1], 3.), .25, .25,
                    np.full(n_centers, .25)]

    def objective(b):
        eta = design @ b
        ridge = penalty * b
        return (np.sum(np.logaddexp(0., eta) - y * eta) + .5 * (b @ ridge),
                design.T @ (expit(eta) - y) + ridge)

    def solve(initial):
        return minimize(objective, initial, jac=True, method='L-BFGS-B',
                        options={'ftol': 1e-12, 'gtol': 1e-7}).x

    initial = np.zeros(design.shape[1])
    center_variance = .04
    if n_centers > 1:
        pilot = solve(initial)
        p = expit(design @ pilot)
        information = design.T @ ((p * (1. - p))[:, None] * design)
        information += np.diag(penalty)
        contrast = np.eye(n_centers) - 1. / n_centers
        cov = np.linalg.solve(information, np.eye(len(pilot))[:, k:])[k:]
        observed = pilot[k:] @ contrast @ pilot[k:]
        noise = np.trace(contrast @ cov)
        # Require evidence above the sampling variance before weakening the
        # penalty; the number of training centers is small.
        center_variance = np.clip((observed - 2. * noise) / (n_centers - 1),
                                  .04, 10.)
        initial = pilot
    penalty[k:] = 1. / center_variance
    return solve(initial)[:k]


def calibrate_multicenter_assay(dataset: dict) -> dict:
    """Return control-aligned query biomarkers and estimated case risks."""
    panels, pooled = _reference_transforms(dataset['reference_controls'])
    train, query = dataset['train'], dataset['query']
    z_train = _align(train, panels, pooled)
    z_query = _align(query, panels, pooled)
    fitted = _fit(z_train, train['stratum'], train['center_id'], np.asarray(train['y']))
    probabilities = expit(_design(z_query, query['stratum']) @ fitted)
    eps = np.finfo(float).eps
    return {'aligned_query_features': z_query,
            'query_probabilities': np.clip(probabilities, eps, 1. - eps),
            'feature_weights': fitted[1:1 + z_query.shape[1]]}
