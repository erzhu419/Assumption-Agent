"""Estimate calibrated two-branch creep from benchtop and chamber coupons."""
import numpy as np
from scipy.optimize import least_squares


def _loads(dataset):
    cal = dataset['calibration_load_cells']
    raw = np.asarray(cal['bridge_raw'], float)
    true = np.asarray(cal['true_channel_loads'], float)
    # Summation commutes with affine least squares: solve directly for axial load.
    return np.linalg.lstsq(np.c_[raw, np.ones(len(raw))], true.sum(axis=1), rcond=None)[0]


def _history(rows, bridge):
    return (rows['past_channel_raw'] @ bridge[:-1] + bridge[-1],
            rows['hold_channel_raw'] @ bridge[:-1] + bridge[-1])


def _branches(history, dt, times, taus):
    past, hold = history
    memory = np.zeros((len(hold), 2))
    for j, width in enumerate(dt):
        gain = -np.expm1(-width / taus)
        memory += gain * (past[:, j, None] - memory)
    amp = hold[:, None] - memory
    curves = amp[:, None, :] * -np.expm1(-times[None, :, None] / taus)
    return np.concatenate((curves, (amp / taus)[:, None, :]), axis=1)


def _observations(rows):
    return np.c_[rows['strain_curves_obs'] - rows['strain_0'][:, None],
                 rows['instant_rate_obs']]


def _spectrum(rows, history, dt, times):
    """Profile one drift rate per coupon out of each weighted spectrum fit."""
    y = _observations(rows)
    t = np.r_[times, 1.0]
    w0 = np.ones_like(y)
    lower = [0.050001, 0.050001, -np.inf, -np.inf]
    upper = [24.999999, 24.999999, np.inf, np.inf]

    def projected(a, w):
        return a - (np.sum(w * a * t, axis=1) / (w @ (t*t)))[:, None] * t

    def residual(p, w):
        b = _branches(history, dt, times, p[:2]) @ p[2:]
        return (np.sqrt(w) * projected(b - y, w)).ravel()

    def refine(p, w):
        return least_squares(residual, p, args=(w,), bounds=(lower, upper),
                             x_scale='jac', max_nfev=75).x

    # Rank starts by robust loss after reducing the leverage of gross curve errors.
    seeds = []
    for fast, slow in ((.22, 1.6), (.55, 4.), (1.25, 12.)):
        taus = np.array([fast, slow]); b = _branches(history, dt, times, taus)
        a = np.stack([projected(b[:, :, i], w0) for i in range(2)], axis=-1)
        coeff = np.linalg.lstsq(a.reshape(-1, 2), projected(y, w0).ravel(), rcond=None)[0]
        p = np.r_[taus, coeff]
        w = w0
        for _ in range(5):
            p = refine(p, w)
            e = residual(p, w).reshape(y.shape) / np.sqrt(w)
            w = 1 / (1 + (e / .018)**2)
        seeds.append((np.log1p((e / .018)**2).sum(), p, w))
    _, p, w = min(seeds, key=lambda item: item[0])
    for _ in range(12):
        p = refine(p, w)
        e = residual(p, w).reshape(y.shape) / np.sqrt(w)
        scale = np.maximum(.003, 1.4826 * np.median(np.abs(e), axis=0))
        # Adjacent time samples measure the same instrument, so pool their scale.
        scale[:-1] = np.median(scale[:-1])
        w = 1 / (1 + (e / scale)**2) / scale**2
    order = np.argsort(p[:2])
    if p[order[1]] - p[order[0]] < 1e-6:
        p[order[0]] = max(.0500001, p[order[1]] - 1e-6)
        p[order[1]] = min(24.9999999, max(p[order[0]] + 1e-6, p[order[1]]))
    diff = y - _branches(history, dt, times, p[:2]) @ p[2:]
    drift = np.sum(w * diff * t, axis=1) / (w @ (t*t))
    return p[:2][order], p[2:][order], drift


def _join(rows, keys):
    return {key: np.concatenate([r[key] for r in rows]) for key in keys}


def _panel(rows):
    panel = _join(rows, ('lab_id', 'ref_coupons'))
    result = {}
    for lab in np.unique(panel['lab_id']):
        ref = panel['ref_coupons'][panel['lab_id'] == lab]
        result[lab] = (np.median(ref.mean(axis=0)), max(1e-8, np.median(ref.std(axis=0))))
    return result


def _align(rows, panel):
    x = np.asarray(rows['coupon_features_raw'], float).copy()
    for lab in np.unique(rows['lab_id']):
        sel = rows['lab_id'] == lab
        center, scale = panel[lab]
        x[sel] = (x[sel] - center) / scale
    return x


def _fill(rows, panel):
    r = _join(rows, ('lab_id', 'stratum', 'coupon_features_raw'))
    x = _align(r, panel)
    _, cells, counts = np.unique(np.c_[r['lab_id'], r['stratum']], axis=0,
                                 return_inverse=True, return_counts=True)
    w = 1.0 / counts[cells, None]
    denom = (np.isfinite(x) * w).sum(axis=0)
    return np.divide(np.nansum(w*x, axis=0), denom, out=np.zeros(x.shape[1]), where=denom > 0)


def _design(rows, panel, fill):
    x = _align(rows, panel)
    return np.c_[np.ones(len(x)), np.where(np.isfinite(x), x, fill)]


def _regress(x, drift, scale, prior=None):
    if prior is None:
        prior = np.linalg.lstsq(x, drift, rcond=None)[0]
    return least_squares(lambda b: (x @ b - drift) / scale, prior,
                         loss='soft_l1').x


def _chamber_drift(x, drift, bench_slopes):
    # The pilot identifies a chamber intercept and gain more reliably than four
    # independent slopes. Do not share the measured chamber spectrum with the bench.
    z = np.c_[np.ones(len(x)), x[:, 1:] @ bench_slopes]
    intercept, gain = np.linalg.lstsq(np.r_[z, [[0., .15]]],
                                      np.r_[drift, .15], rcond=None)[0]
    return np.r_[intercept, gain * bench_slopes]


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    pilot = dataset.get('chamber_verification_trials')
    dt, times = map(np.asarray, (dataset['past_dt'], dataset['eval_times']))
    bridge = _loads(dataset)
    taus, weights, drift = _spectrum(train, _history(train, bridge), dt, times)
    panel = _panel([train, test])
    fill = _fill([train, test], panel)
    beta = _regress(_design(train, panel, fill), drift, .009)
    if pilot is not None and len(pilot['lab_id']):
        taus, weights, drift = _spectrum(pilot, _history(pilot, bridge), dt, times)
        panel = _panel([test, pilot])
        # A training-only donor pool keeps the sparse pilot from setting the
        # missing-feature values used to fit its own drift coefficients.
        fill = _fill([train], {**_panel([train]), **panel})
        beta = _chamber_drift(_design(pilot, panel, fill), drift, beta[1:])
    b = _branches(_history(test, bridge), dt, times, taus) @ weights
    base = _design(test, panel, fill) @ beta
    return {'estimated_taus': taus,
            'hereditary_weights': np.r_[weights, beta],
            'predicted_strain': test['strain_0'][:, None] + b[:, :-1] + base[:, None] * times,
            'initial_rate_pred': b[:, -1] + base}
