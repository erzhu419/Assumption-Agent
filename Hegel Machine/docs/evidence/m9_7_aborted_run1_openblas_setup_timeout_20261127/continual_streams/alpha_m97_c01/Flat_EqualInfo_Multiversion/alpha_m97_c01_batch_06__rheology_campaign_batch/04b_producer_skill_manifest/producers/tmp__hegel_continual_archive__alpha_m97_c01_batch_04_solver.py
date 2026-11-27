"""Calibrated two-branch creep with a damage-gated crystallization plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _channel_points(rows, channel):
    x, y = (np.asarray(rows[k], dtype=float) for k in ('cmd_stress', 'true_stress'))
    use = (np.asarray(rows['lab_id']) == channel) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _clean(x, y):
    if len(x) < 3 or not np.ptp(x):
        return np.ones(len(x), dtype=bool)
    dx, dy = x[:, None] - x, y[:, None] - y
    slope = np.median([np.median(b[a != 0] / a[a != 0])
                       for a, b in zip(dx, dy) if np.any(a != 0)])
    intercepts = y - slope*x
    return np.abs(intercepts - np.median(intercepts)) <= 3.0


def _calibrate(rows):
    result = np.empty((4, 2))
    sweeps = None
    for c in range(4):
        x, y = _channel_points(rows, c)
        clean = _clean(x, y)
        if clean.sum() >= 2 and np.ptp(x[clean]) and (clean.sum() >= 4 or clean.all()):
            result[c] = np.polyfit(x[clean], y[clean], 1)
            continue
        if len(x) >= 2 and clean.all():
            raise ValueError(f'Calibration commands have no variation for channel {c}')
        if sweeps is None:
            sweeps = json.loads(_ARCHIVE.read_text())
        for sweep in reversed(sweeps):
            a, b = _channel_points(sweep, c)
            keep = _clean(a, b)
            if keep.sum() >= 2 and np.ptp(a[keep]):
                result[c] = np.polyfit(a[keep], b[keep], 1)
                break
        else:
            raise ValueError(f'No usable prefix calibration for channel {c}')
    return result


def _prepare(dataset):
    def join(key):
        return np.concatenate((dataset['train'][key], dataset['test'][key]))

    lab, stratum = join('lab_id'), join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    labs = np.unique(lab)
    for c in labs:
        use = lab == c
        panel = ref[use]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        spread = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[use] = (x[use] - center)/spread
    cells = [(lab == c) & (stratum == s) for c, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    for j in range(x.shape[1]):
        observed = np.isfinite(x[:, j])
        means = [np.mean(x[use & observed, j]) if np.any(use & observed) else np.nan for use in cells]
        for use, mean in zip(cells, means):
            if np.any(use & ~observed):
                x[use & ~observed, j] = mean if np.isfinite(mean) else np.nanmean(means)
    excess = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)
    damage = (excess**2) @ np.asarray(dataset['past_dt'])
    plateau = np.full(len(hold), dataset['shear_banding_index'] <= .50)
    # Banding selects the baseline flow basis; damage controls only the network terms.
    return past, hold, x, plateau, np.where(plateau, np.tanh(damage), 0.)


def _design(tau, state, dt, time):
    past, hold, x, plateau, act = state
    z = np.zeros((len(hold), 2))
    decay = np.exp(-dt[:, None]/tau)
    step = -np.expm1(-dt[:, None]/tau)
    for j in range(len(dt)):
        z = decay[j]*z + step[j]*past[:, j, None]
    gap = hold[:, None] - z
    creep = -np.expm1(-time[:, None]/tau)
    flow = np.where(plateau[:, None], np.log1p(time), time)
    drift = np.column_stack((np.ones(len(hold)), x))
    net, memory = act*hold, act*z[:, 0]
    curves = np.concatenate((gap[:, None, :]*creep[None, :, :],
                             drift[:, None, :]*flow[:, :, None],
                             (net[:, None]*creep[:, 0])[:, :, None],
                             (memory[:, None]*flow)[:, :, None]), axis=2)
    rates = np.column_stack((gap/tau, drift, net/tau[0], memory))
    return curves, rates


def _mad(x, axis=None):
    return 1.4826*np.nanmedian(np.abs(x - np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def _fit(state, dt, time, delta, rate):
    cm, rm = np.isfinite(delta), np.isfinite(rate)
    y = np.r_[delta[cm], rate[rm]]
    n_curve = cm.sum()
    w = np.r_[np.ones(n_curve), np.full(rm.sum(), time[0])]
    lo = max(.02*np.min(dt[dt > 0]), 1e-4)
    hi = 12*(dt.sum()+time.max())
    bounds = np.log([[lo, lo], [hi, hi]])

    def unpack(p):
        t = np.exp(p)
        return np.array([t[0], t.sum()])

    def profile(p):
        a, b = _design(unpack(p), state, dt, time)
        a = np.concatenate((a[cm], b[rm]))
        weighted_a = a*w[:, None]
        coef = np.linalg.lstsq(weighted_a, y*w, rcond=None)[0]
        return a @ coef-y, coef

    def opt(p):
        return least_squares(lambda q: profile(q)[0]*w, p, bounds=bounds,
                             max_nfev=100, ftol=1e-9)

    starts = np.clip(np.log([[.25, 2.25], [.65, 3.35], [1.25, 6.75], [.15, 8.85]]), *bounds)
    p = min((opt(s) for s in starts), key=lambda fit: fit.cost).x
    for _ in range(8):
        residual, _ = profile(p)
        e = np.full(delta.shape, np.nan)
        e[cm] = residual[:n_curve]
        sigma_c = np.maximum(_mad(e, axis=0), max(.35*_mad(e), 1e-7))
        sigma_r = max(_mad(residual[n_curve:]), 1e-7) if rm.any() else 1.
        sigma = np.r_[np.broadcast_to(sigma_c, delta.shape)[cm], np.full(rm.sum(), sigma_r)]
        w = 1/np.hypot(sigma, residual/2.38)
        p = opt(p).x
    return unpack(p), profile(p)[1]


def solve_alpha_m97_c01_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    time, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    state = _prepare(dataset)
    n = len(train['lab_id'])
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, coef = _fit(tuple(a[:n] for a in state), dt, time, delta, train['instant_rate_obs'])
    curve, rate = _design(tau, tuple(a[n:] for a in state), dt, time)
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ coef,
            'initial_rate_pred': rate @ coef}
