"""Calibrate multi-wing coupon stress and fit a thresholded Prony law."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _rows(panel, lab):
    x, y = (np.asarray(panel[k], dtype=float) for k in ('cmd_stress', 'true_stress'))
    keep = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[keep], y[keep]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _local_fit(x, y, prior):
    i, j = np.triu_indices(len(x), 1)
    span = x[j] - x[i]
    use = np.abs(span) >= 0.12
    slope = (y[j[use]] - y[i[use]]) / span[use]
    lines = np.column_stack((slope, y[i[use]] - slope * x[i[use]]))
    models = np.vstack((prior, lines[slope > 0]))
    err = np.abs(y - models[:, :1] * x - models[:, 1:])
    good = err <= 0.003
    count = good.sum(axis=1)
    loss = np.minimum((err / 0.003)**2, 1).sum(axis=1)
    best = np.lexsort((loss, -count))[0]
    return models[best], good[best]


def _calibrate(panel):
    # Compare complete campaign regimes before borrowing parameters across labs.
    # Thus a sparse check on one wing selects the matching sweep for the other.
    identity = np.column_stack((np.ones(4), np.zeros(4)))
    history = json.loads(_HISTORY.read_text()) if _HISTORY.is_file() else []
    sweeps = []
    for old in reversed(history):
        sweep = []
        for lab in range(4):
            x, y = _rows(old, lab)
            _, inliers = _local_fit(x, y, identity[lab])
            if inliers.sum() < 3 or np.ptp(x[inliers]) < 0.8:
                break
            sweep.append(_line(x[inliers], y[inliers]))
        if len(sweep) == 4:
            sweeps.append(sweep)
    sweeps = np.asarray(sweeps + [identity])
    rows = [_rows(panel, lab) for lab in range(4)]
    error = np.concatenate([
        np.abs(y - sweeps[:, lab, :1] * x - sweeps[:, lab, 1:])
        for lab, (x, y) in enumerate(rows)
    ], axis=1)
    count = (error <= 0.003).sum(axis=1)
    loss = np.minimum((error / 0.003)**2, 1).sum(axis=1)
    prior = sweeps[np.lexsort((loss, -count))[0]].copy()
    for lab, (x, y) in enumerate(rows):
        _, inliers = _local_fit(x, y, prior[lab])
        n = inliers.sum()
        if n >= 3 and np.ptp(x[inliers]) >= 0.8:
            prior[lab] = _line(x[inliers], y[inliers])
        elif len(x) == 2 and n == 2 and np.ptp(x) >= 0.8:
            # Keep the archived fit if both checks agree; otherwise a coherent
            # pair with no matching archive is the only local affine estimate.
            err = np.abs(y - (prior[lab, 0] * x + prior[lab, 1]))
            if np.all(err > 0.003):
                prior[lab] = _line(x, y)
    return prior


def _align(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    strata = np.r_[train['stratum'], test['stratum']]
    refs = np.concatenate([train['ref_coupons'], test['ref_coupons']])
    raw = np.concatenate([train['coupon_features_raw'], test['coupon_features_raw']])
    aligned = np.empty_like(raw, dtype=float)
    for c in np.unique(labs):
        lab = labs == c
        r = refs[lab]
        r = r[:, np.isfinite(r).any(axis=0)]
        center = np.median(np.nanmean(r, axis=0))
        spread = np.median(np.nanstd(r, axis=0))
        x = (raw[lab] - center) / spread
        z = strata[lab]
        cells = np.unique(z)
        means = np.full((len(cells), x.shape[1]), np.nan)
        for i, s in enumerate(cells):
            block = x[z == s]
            count = np.isfinite(block).sum(axis=0)
            np.divide(np.nansum(block, axis=0), count, out=means[i], where=count > 0)
        counts = np.isfinite(means).sum(axis=0)
        fallback = np.divide(np.nansum(means, axis=0), counts, out=np.zeros(x.shape[1]), where=counts > 0)
        for i, s in enumerate(cells):
            take = z == s
            x[take] = np.where(np.isfinite(x[take]), x[take], np.where(np.isfinite(means[i]), means[i], fallback))
        aligned[lab] = x
    return aligned[:len(train['lab_id'])], aligned[len(train['lab_id']):]


def _stress(split, affine):
    gain, bias = affine[np.asarray(split['lab_id'])].T
    return gain[:, None] * split['past_stress_cmd'] + bias[:, None], gain * split['stress_hold_cmd'] + bias


def _prepare(dt, history, hold, aligned, yield_stress):
    after = np.cumsum(dt[::-1])[::-1] - dt
    drift = np.column_stack((np.ones(len(hold)), aligned))
    excess = np.maximum(np.abs(history) - yield_stress, 0.)
    plastic_history = (excess**2 @ dt) * hold
    plastic_hold = np.maximum(np.abs(hold) - yield_stress, 0.)**2 * np.sign(hold)
    return dt, after, history, hold, drift, plastic_hold, plastic_history


def _design(taus, times, state):
    dt, after, hist, hold, drift, plastic_hold, plastic_history = state
    memory = np.exp(-after[:, None] / taus) * (-np.expm1(-dt[:, None] / taus))
    delta = hold[:, None] - np.einsum('ns,sm->nm', hist, memory)
    growth = -np.expm1(-times[:, None] / taus)
    rate = np.column_stack((delta / taus, drift, plastic_hold, plastic_history / taus[0]))
    curves = np.concatenate((delta[:, None, :] * growth,
                             drift[:, None, :] * times[None, :, None],
                             plastic_hold[:, None, None] * times[None, :, None],
                             plastic_history[:, None, None] * growth[None, :, :1]), axis=2)
    return curves, rate


def _fit(times, state, strain, rate):
    n, k = strain.shape
    target = np.r_[strain.ravel(), rate]
    active = np.r_[np.ones(7, dtype=bool), [np.any(v) for v in state[-2:]]]

    def build(taus):
        a, b = _design(taus, times, state)
        return np.concatenate((a.reshape(n*k, 9), b))[:, active]

    # Use a fraction of the remaining range for the slow branch to enforce order.
    lower, upper = 0.2, 8.
    p_min = np.r_[lower, 1e-6, np.full(active.sum(), -np.inf)]
    p_max = np.r_[upper - 1e-6, 1., np.full(active.sum(), np.inf)]

    def taus(q):
        return np.array([q[0], q[0] + (upper - q[0]) * q[1]])

    def optimize(start, scales, loss):
        def residual(q):
            return (np.einsum('ij,j->i', build(taus(q)), q[2:]) - target) / scales
        return least_squares(residual, start, bounds=(p_min, p_max),
                             loss=loss, x_scale='jac', max_nfev=160)

    pilots = []
    for a, b in ((0.35, 1.3), (0.65, 3.), (1.15, 5.4)):
        w = np.linalg.lstsq(build([a, b]), target, rcond=None)[0]
        pilots.append(optimize(np.r_[a, (b-a)/(upper-a), w], 0.05, 'soft_l1'))
    pilot = min(pilots, key=lambda fit: fit.cost)
    raw = 0.05 * pilot.fun

    def mad(v):
        return max(1e-6, 1.4826 * np.median(np.abs(v - np.median(v))))

    scales = np.r_[np.full(n*k, mad(raw[:n*k])), np.full(n, mad(raw[n*k:]))]
    fit = optimize(pilot.x, 2.385 * scales, 'cauchy')
    weights = np.zeros(9)
    weights[active] = fit.x[2:]
    return taus(fit.x), weights


def solve_alpha_w04_batch_07(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    affine = _calibrate(dataset['calibration_load_cells'])
    x, xt = _align(train, test)
    yld = dataset['yield_stress']
    tr = _prepare(dt, *_stress(train, affine), x, yld)
    te = _prepare(dt, *_stress(test, affine), xt, yld)
    strain = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate_obs = train['instant_rate_obs']
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    # Missing features add an unknown drift to the calibration observations.
    # Keep those rows if excluding them would remove support for a yield term.
    if complete.sum() >= 2 * x.shape[1] and all(not np.any(v) or np.any(v[complete]) for v in tr[-2:]):
        tr_fit = tr[:2] + tuple(v[complete] for v in tr[2:])
        taus, weights = _fit(times, tr_fit, strain[complete], rate_obs[complete])
    else:
        taus, weights = _fit(times, tr, strain, rate_obs)
    curves, rate = _design(taus, times, te)
    predicted_rate = rate @ weights
    predicted_strain = test['strain_0'][:, None] + curves @ weights

    # Keep the cell-mean feature alignment. When a feature is missing, the
    # observed onset rate can resolve its unidentified time-linear drift in
    # the curve forecast. The independently predicted onset rate stays unchanged.
    missing = ~np.isfinite(test['coupon_features_raw'])
    if 'instant_rate_obs' in test:
        observed = np.asarray(test['instant_rate_obs'])
        use = missing.any(axis=1) & np.isfinite(observed)
        if use.any():
            # Bound the update by the empirical variation of the omitted
            # feature contribution, not by the magnitude of the observation.
            centered = x - x.mean(axis=0)
            contribution = missing[use] * weights[3:7]
            limit = 3. * np.std(centered @ contribution.T, axis=0)
            update = np.clip(observed[use] - predicted_rate[use], -limit, limit)
            predicted_strain[use] += update[:, None] * times

    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'viscoplastic_weights': weights[7:], 'predicted_strain': predicted_strain,
            'initial_rate_pred': predicted_rate}
