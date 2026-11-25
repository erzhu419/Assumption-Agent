"""Reference-panel alignment and stratum-adjusted assay probabilities."""

import numpy as np
from sklearn.linear_model import LogisticRegression


def _panel_statistics(block):
    """Estimate per-biomarker location and log scale from finite controls."""
    n = np.zeros(block.shape[1], dtype=int)
    loc = np.zeros(block.shape[1])
    log_scale = np.zeros(block.shape[1])
    usable = np.zeros(block.shape[1], dtype=bool)
    for j, col in enumerate(block.T):
        col = col[np.isfinite(col)]
        n[j] = col.size
        if col.size >= 2:
            sd = col.std(ddof=1)
            if sd > 0:
                loc[j] = col.mean()
                log_scale[j] = np.log(sd)
                usable[j] = True
    return n, loc, log_scale, usable


def _shrink(estimates, variance, anchor):
    # Estimate real between-biomarker variation after subtracting sampling noise.
    # Two observations suffice for the fallback contract, but are too noisy
    # to inform how strongly the other biomarkers should be pooled.
    precision = 1 / variance
    mean = np.average(estimates, weights=precision)
    noise = precision.size - 1
    spread = np.dot(precision, (estimates - mean) ** 2)
    effective_precision = precision.sum() - np.dot(precision, precision) / precision.sum()
    signal = max(0.0, (spread - noise) / effective_precision) if effective_precision > 0 else 0.0
    return anchor + (estimates - anchor) * (signal / (signal + variance))


def _reference_transforms(reference):
    centers = np.asarray(reference['center_id'])
    controls = np.asarray(reference['X'], dtype=float)
    summaries = {c: _panel_statistics(controls[centers == c]) for c in np.unique(centers)}
    transforms = {}
    for center, (n, loc, log_scale, ok) in summaries.items():
        if not np.any(ok):
            continue
        reliable = ok & (n >= 4)
        anchor = reliable if np.any(reliable) else ok
        loc0 = np.median(loc[anchor])
        scale0 = np.median(log_scale[anchor])
        means = np.full(loc.shape, loc0)
        scales = np.full(log_scale.shape, scale0)
        means[reliable] = _shrink(loc[reliable], np.exp(2 * scale0) / n[reliable], loc0)
        scales[reliable] = _shrink(log_scale[reliable], 0.5 / (n[reliable] - 1), scale0)
        transforms[center] = means, np.exp(scales)
    if transforms:
        locs, scales = zip(*transforms.values())
        fallback = np.median(locs, axis=0), np.median(scales, axis=0)
    else:
        _, loc, log_scale, _ = _panel_statistics(controls)
        fallback = loc, np.exp(log_scale)
    return transforms, fallback


def _align(part, transforms, fallback):
    values = np.asarray(part['X'], dtype=float)
    center = np.asarray(part['center_id'])
    aligned = np.empty_like(values)
    for c in np.unique(center):
        rows = center == c
        loc, scale = transforms.get(c, fallback)
        aligned[rows] = (values[rows] - loc) / scale
    aligned[~np.isfinite(aligned)] = 0.0
    return aligned


def calibrate_multicenter_assay(dataset: dict) -> dict:
    """Fit training outcomes and predict at the average training-center effect."""
    train, query = dataset['train'], dataset['query']
    transforms, fallback = _reference_transforms(dataset['reference_controls'])
    xt = _align(train, transforms, fallback)
    xq = _align(query, transforms, fallback)
    strata = np.unique(train['stratum'])
    st = np.asarray(train['stratum'])[:, None] == strata
    sq = np.asarray(query['stratum'])[:, None] == strata
    center = np.asarray(train['center_id'])
    centers = np.unique(center)
    ct = (center[:, None] == centers).astype(float)
    ct -= ct.mean(axis=0)
    # Keep observed class priors. Center intercepts absorb residual training-site
    # effects; no such intercept is estimated from an unlabeled query center.
    # Penalize nuisance intercepts less than transferable biomarker effects.
    model = LogisticRegression(C=0.3, max_iter=1000)
    model.fit(np.column_stack((xt, st, 3.0 * ct)), train['y'])
    pq = model.predict_proba(np.column_stack((xq, sq, np.zeros((len(xq), len(centers))))))[:, 1]
    return {
        'aligned_query_features': xq,
        'query_probabilities': np.clip(pq, np.finfo(float).eps, 1 - np.finfo(float).eps),
        'feature_weights': model.coef_[0, :xt.shape[1]].copy(),
    }
