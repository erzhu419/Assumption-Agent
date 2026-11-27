"""Fit a two-branch strain-crystallization model in physical stress coordinates."""
import json

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _points(sweep, channel):
    lab, x, y = (np.asarray(sweep[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    use = (lab == channel) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _calibration(dataset):
    history = None
    fits = np.empty((4, 2))
    for c in range(4):
        x, y = _points(dataset['calibration_load_cells'], c)
        enough = len(x) >= 2 and np.ptp(x) > 0
        emi = False
        if len(x) >= 3:
            for j in range(len(x)):
                use = np.arange(len(x)) != j
                if np.ptp(x[use]) > 0:
                    a, b = _line(x[use], y[use])
                    emi |= abs(y[j] - (a*x[j]+b)) > 3.0
        if enough and not emi:
            fits[c] = _line(x, y)
            continue
        if history is None:
            with open(_ARCHIVE) as f:
                history = json.load(f)
        for sweep in reversed(history):
            u, v = _points(sweep, c)
            if len(u) >= 2 and np.ptp(u) > 0:
                a, b = _line(u, v)
                if np.max(np.abs(v - (a*u+b))) < 0.02:
                    break
        else:
            raise ValueError(f'No clean archived sweep for channel {c}')
        clean = np.abs(y - (a*x+b)) <= 3.0 if emi else np.ones(len(x), dtype=bool)
        x, y = x[clean], y[clean]
        fits[c] = _line(x, y) if len(x) >= 2 and np.ptp(x) > 0 else (a, np.mean(y-a*x) if len(x) else b)
    return fits


def _aligned(train, test):
    def join(k):
        return np.concatenate((train[k], test[k]))
    lab, stratum, ref, raw = (join(k) for k in ('lab_id', 'stratum', 'ref_coupons', 'coupon_features_raw'))
    x = raw.astype(float, copy=True)
    cells = {}
    for c in np.unique(lab):
        rows = lab == c
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[rows] = (x[rows] - center) / scale
        for s in np.unique(stratum[rows]):
            mask = rows & (stratum == s)
            cells[c, s] = (mask, np.nanmean(x[mask], axis=0))
    for (c, s), (rows, mean) in cells.items():
        if not np.isfinite(mean).all():
            peers = [v for (lab_id, group), (_, v) in cells.items() if group == s]
            mean = np.where(np.isfinite(mean), mean, np.nanmean(peers, axis=0))
        x[rows] = np.where(np.isfinite(x[rows]), x[rows], mean)
    return np.split(x, [len(train['lab_id'])])


def _inputs(data, x, calibration, dt, crit, band):
    gain, bias = calibration[np.asarray(data['lab_id'])].T
    past = data['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = data['stress_hold_cmd'] * gain + bias
    damage = (np.maximum(np.abs(past)-crit, 0.0)**2) @ dt
    act = np.tanh(damage) if band <= 0.50 else np.zeros_like(damage)
    return past, hold, x, act


def _design(tau, dt, times, inputs):
    past, hold, x, act = inputs
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = np.exp(-remaining[:, None]/tau) * (-np.expm1(-dt[:, None]/tau))
    z = past @ kernel
    gap = hold[:, None] - z
    transient = -np.expm1(-times[:, None]/tau)
    drift = np.column_stack((np.ones(len(x)), x))
    coupling = act * z[:, 0]
    add = act * hold
    curve = np.empty((len(x), len(times), 9))
    curve[:, :, :2] = gap[:, None, :] * transient
    logt = np.log1p(times)
    curve[:, :, 2:7] = drift[:, None, :] * logt[None, :, None]
    curve[:, :, 7] = add[:, None] * transient[:, 0]
    curve[:, :, 8] = coupling[:, None] * logt
    rate = np.column_stack((gap / tau, drift, add / tau[0], coupling))
    return curve, rate


def _fit(dt, times, inputs, curves, rates):
    target = np.r_[curves.ravel(), rates]
    nc = curves.size

    def design(p):
        a, b = _design(np.exp(p), dt, times, inputs)
        return np.vstack((a.reshape(-1, 9), b))

    def profile(p, scale, robust):
        a = design(p) / scale[:, None]
        y = target / scale
        w = np.ones(len(y))
        for _ in range(12 if robust else 1):
            at = a.T * w
            mat = np.einsum('ij,jk->ik', at, a, optimize=False)
            rhs = np.einsum('ij,j->i', at, y)
            try:
                beta = np.linalg.solve(mat, rhs)
            except np.linalg.LinAlgError:
                root = np.sqrt(w)
                beta = np.linalg.lstsq(a * root[:, None], y * root, rcond=None)[0]
            r = np.einsum('ij,j->i', a, beta) - y
            w = 1 / (1 + (r / 2.5)**2)
        objective = np.sign(r) * 2.5 * np.sqrt(np.log1p((r/2.5)**2)) if robust else r
        return objective, beta, r*scale

    def optimize(start, scale, robust):
        return least_squares(lambda p: profile(p, scale, robust)[0], start,
                             bounds=(np.log([0.06, 0.7]), np.log([1.25, 25.])),
                             diff_step=1e-5, max_nfev=80, ftol=1e-8, xtol=1e-8, gtol=1e-8)

    def scales(r):
        r = r[:nc].reshape(curves.shape), r[nc:]
        mad = [1.4826 * np.median(np.abs(v - np.median(v, axis=0)), axis=0) for v in r]
        return np.maximum(np.r_[np.tile(mad[0], len(curves)), np.full(len(rates), mad[1])], 1e-5)

    scale = np.r_[np.ones(nc), np.full(len(rates), 1.5)]
    starts = np.log([[0.22, 1.9], [0.57, 4.0], [0.96, 9.0]])
    init = min((optimize(p, scale, False) for p in starts), key=lambda r: r.cost)
    scale = scales(profile(init.x, scale, False)[2])
    r = optimize(init.x, scale, True)
    scale = scales(profile(r.x, scale, True)[2])
    r = optimize(r.x, scale, True)
    return np.exp(r.x), profile(r.x, scale, True)[1]


def solve_alpha_m97_c02_batch_07(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    calibration = _calibration(dataset)
    args = (calibration, dt, dataset['crit_stress'], dataset['shear_banding_index'])
    xt, xe = _aligned(train, test)
    measured = np.isfinite(train['coupon_features_raw']).all(axis=1)
    inputs = tuple(v[measured] for v in _inputs(train, xt, *args))
    tau, weights = _fit(dt, times, inputs,
                        (train['strain_curves_obs'] - train['strain_0'][:, None])[measured],
                        train['instant_rate_obs'][measured])
    curve, rate = _design(tau, dt, times, _inputs(test, xe, *args))
    return {'estimated_taus': tau, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
