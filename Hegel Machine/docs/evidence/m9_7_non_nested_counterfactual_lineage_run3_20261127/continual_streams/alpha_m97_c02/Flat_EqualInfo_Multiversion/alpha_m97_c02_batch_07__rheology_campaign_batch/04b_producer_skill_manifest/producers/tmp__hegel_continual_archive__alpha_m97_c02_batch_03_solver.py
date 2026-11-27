"""Calibrated two-branch Prony model with jammed-slip terminal flow."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _channel(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    use = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _historical(lab):
    if _ARCHIVE.is_file():
        for sweep in reversed(json.loads(_ARCHIVE.read_text())):
            x, y = _channel(sweep, lab)
            if len(x) >= 2 and np.ptp(x) > 0:
                return _affine(x, y)
    return None


def _calibrate(sweep):
    calibration = np.empty((4, 2))
    for lab in range(4):
        x, y = _channel(sweep, lab)
        clean = np.ones(len(x), dtype=bool)
        if len(x) >= 3 and np.ptp(x) > 0:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[j] != x[i]
            slopes = (y[j[distinct]] - y[i[distinct]]) / (x[j[distinct]] - x[i[distinct]])
            gain = np.median(slopes)
            offset = np.median(y - gain*x)
            clean = np.abs(y - gain*x - offset) <= 3.0
            # With one transient in a small sweep, a fit that includes
            # the spike can mask it. Select the most self-consistent
            # leave-one-out sweep before testing its omitted row.
            if len(x) >= 4:
                best_error = np.inf
                omitted = None
                for k in range(len(x)):
                    keep = np.arange(len(x)) != k
                    if np.ptp(x[keep]) == 0:
                        continue
                    g, b = _affine(x[keep], y[keep])
                    error = np.max(np.abs(y[keep] - (g*x[keep] + b)))
                    if error < best_error:
                        best_error = error
                        omitted = k
                        deleted_error = abs(y[k] - (g*x[k] + b))
                if omitted is not None and best_error <= 3.0 < deleted_error:
                    clean = np.arange(len(x)) != omitted
        if len(x) < 2 or (not clean.all() and clean.sum() < 4):
            archived = _historical(lab)
            if archived is not None:
                calibration[lab] = archived
                continue
        if clean.sum() < 2 or np.ptp(x[clean]) == 0:
            raise ValueError(f'No identifiable load-cell calibration for channel {lab}')
        calibration[lab] = _affine(x[clean], y[clean])
    return calibration


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def both(key):
        return np.concatenate((train[key], test[key]))

    labs, strata = both('lab_id'), both('stratum')
    raw, refs = both('coupon_features_raw'), both('ref_coupons')
    x = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        use = labs == lab
        panel = refs[use]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (raw[use] - center) / scale
    _, cell = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = []
    for index in np.unique(cell):
        values = x[cell == index]
        counts = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(x.shape[1], np.nan), where=counts != 0))
    x = np.where(np.isfinite(x), x, np.nanmean(means, axis=0))
    gains, biases = _calibrate(dataset['calibration_load_cells'])[labs].T
    past = gains[:, None] * both('past_stress_cmd') + biases[:, None]
    hold = gains * both('stress_hold_cmd') + biases
    return past, hold, x


def _design(past, hold, x, dt, times, taus):
    # Unroll the relaxation recurrence: each increment decays for the
    # duration of all the steps that follow it.
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = -np.expm1(-dt[:, None]/taus) * np.exp(-remaining[:, None]/taus)
    gap = hold[:, None] - past @ kernel
    relaxation = -np.expm1(-times[:, None]/taus)
    phi = times / (1.0 + 0.85*times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    delta = gap[:, 0] - gap[:, 1]
    flow = np.column_stack((np.ones_like(hold), x, slip))
    curve = np.concatenate((gap[:, None, :]*relaxation[None, :, :],
                            flow[:, None, :]*phi[None, :, None],
                            delta[:, None, None]*relaxation[None, :, :1]), axis=2)
    rate = np.column_stack((gap/taus, flow, delta/taus[0]))
    return curve, rate


def _mad(x):
    return max(1e-7, 1.4826*np.median(np.abs(x-np.median(x))))


def _fit(past, hold, x, dt, times, curves, rates):
    cm, rm = np.isfinite(curves), np.isfinite(rates)
    target = np.r_[curves[cm], rates[rm]]
    nc = cm.sum()
    # Positive separation keeps branch 0 fast in both the profile and the
    # robust fit; the asymmetric slip term is not invariant to a branch swap.
    lower, upper = np.log([.015, .015]), np.log([20., 60.])

    def unpack(q):
        fast, separation = np.exp(q[:2])
        return np.array([fast, fast + separation])

    def design(q):
        c, r = _design(past, hold, x, dt, times, unpack(q))
        return np.concatenate((c[cm], r[rm]))

    scale = np.r_[np.ones(nc), np.full(rm.sum(), 2.)]

    def profile(q, ret_weights=False):
        a = design(q)/scale[:, None]
        b = target/scale
        w = np.linalg.lstsq(a, b, rcond=None)[0]
        return w if ret_weights else np.einsum('ij,j->i', a, w) - b

    starts = ((.15, .7), (.4, 2.), (.9, 3.5), (1.8, 7.))
    best = min((least_squares(profile, np.log([f, s-f]), bounds=(lower, upper),
                              max_nfev=75) for f, s in starts), key=lambda r: r.cost)
    p = np.r_[best.x, profile(best.x, True)]

    def residual(q):
        return np.einsum('ij,j->i', design(q), q[2:]) - target

    bounds = (np.r_[lower, np.full(9, -np.inf)], np.r_[upper, np.full(9, np.inf)])
    for _ in range(2):
        e = residual(p)
        noise = np.r_[np.full(nc, _mad(e[:nc])), np.full(rm.sum(), _mad(e[nc:]))]
        p = least_squares(lambda q: residual(q)/noise, p, bounds=bounds,
                          loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=100).x
    return unpack(p), p[2:]


def solve_alpha_m97_c02_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(*(a[:n][complete] for a in data), dt, times,
                         curves[complete], train['instant_rate_obs'][complete])
    curve, rate = _design(*(a[n:] for a in data), dt, times, taus)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
