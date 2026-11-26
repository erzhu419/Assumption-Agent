"""Calibrate polymer coupon data and fit a two-branch creep law."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def _rows(panel, lab):
    x = np.asarray(panel['cmd_stress'], dtype=float)
    y = np.asarray(panel['true_stress'], dtype=float)
    m = (np.asarray(panel['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _consensus(x, y, priors):
    i, j = np.triu_indices(len(x), 1)
    use = np.abs(x[j] - x[i]) >= 0.12
    i, j = i[use], j[use]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    pairs = np.column_stack((slopes, y[i] - slopes * x[i]))
    models = np.vstack((priors, pairs[pairs[:, 0] > 0]))
    err = np.abs(y - models[:, :1] * x - models[:, 1:])
    inside = err <= 0.003
    n = inside.sum(axis=1)
    loss = np.minimum((err / 0.003)**2, 1).sum(axis=1)
    is_archive = np.arange(len(models)) < len(priors)
    # Prefer an existing affine regime over a two-point line when both explain
    # the same sparse checks. Full sweeps instead compete on residuals.
    best = np.lexsort((loss, (n < 3) & ~is_archive, -n))[0]
    return models[best], inside[best]


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _calibrate(panel):
    identity = np.array([[1., 0.]])
    path = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')
    history = None
    output = []
    for lab in range(4):
        x, y = _rows(panel, lab)
        estimate, m = _consensus(x, y, identity)
        if m.sum() >= 3 and np.ptp(x[m]) >= 0.8:
            output.append(_line(x[m], y[m]))
            continue
        if history is None:
            history = json.loads(path.read_text()) if path.is_file() else []
        priors = []
        for old in reversed(history):
            a, b = _rows(old, lab)
            _, good = _consensus(a, b, identity)
            if good.sum() >= 3 and np.ptp(a[good]) >= 0.8:
                priors.append(_line(a[good], b[good]))
        choices = np.vstack((priors, identity)) if priors else identity
        estimate, m = _consensus(x, y, choices)
        if not m.any() and len(x):
            err = np.abs(y - choices[:, :1] * x - choices[:, 1:])
            estimate = choices[np.argmin(np.median(err, axis=1))]
        elif m.sum() >= 2 and np.ptp(x[m]) >= 0.12 and (m.sum() >= 3 or not np.any(np.all(choices == estimate, axis=1))):
            estimate = _line(x[m], y[m])
        output.append(estimate)
    return np.asarray(output)

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


def solve_alpha_w04_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    affine = _calibrate(dataset['calibration_load_cells'])
    x, xt = _align(train, test)
    yld = dataset['yield_stress']
    tr = _prepare(dt, *_stress(train, affine), x, yld)
    te = _prepare(dt, *_stress(test, affine), xt, yld)
    taus, weights = _fit(times, tr, train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs'])
    curves, rate = _design(taus, times, te)
    predicted_rate = rate @ weights
    predicted_strain = test['strain_0'][:, None] + curves @ weights

    # Keep the cell-mean feature alignment. When a feature is missing, the
    # observed onset rate can resolve its unidentified time-linear drift.
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
            predicted_rate[use] += update
            predicted_strain[use] += update[:, None] * times

    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'viscoplastic_weights': weights[7:], 'predicted_strain': predicted_strain,
            'initial_rate_pred': predicted_rate}
