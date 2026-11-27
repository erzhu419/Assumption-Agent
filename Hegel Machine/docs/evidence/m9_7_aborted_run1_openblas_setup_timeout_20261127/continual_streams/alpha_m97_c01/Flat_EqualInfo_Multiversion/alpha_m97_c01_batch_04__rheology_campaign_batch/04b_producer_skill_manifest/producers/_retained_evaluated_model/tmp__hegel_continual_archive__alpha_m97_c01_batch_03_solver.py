"""Calibrated two-branch rheology in the jammed-slip regime."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _calibrate(rows):
    cmd, truth = (np.asarray(rows[k], dtype=float) for k in ('cmd_stress', 'true_stress'))
    lab = np.asarray(rows['lab_id'])
    result = np.empty((4, 2))
    sweeps = None
    for c in range(4):
        mask = (lab == c) & np.isfinite(cmd) & np.isfinite(truth)
        x, y = cmd[mask], truth[mask]
        spikes = False
        if len(x) >= 3 and np.ptp(x) > 0:
            dx, dy = x[:, None] - x, y[:, None] - y
            slope = np.median([np.median(b[a != 0] / a[a != 0])
                               for a, b in zip(dx, dy) if np.any(a != 0)])
            clean = np.abs(y - slope*x - np.median(y - slope*x)) <= 3.0
            spikes = not clean.all()
            x, y = x[clean], y[clean]
        usable = len(x) >= 2 and np.ptp(x) > 0
        if usable and (len(x) >= 4 or not spikes):
            result[c] = np.polyfit(x, y, 1)
            continue
        if len(x) < 2 or spikes:
            if sweeps is None:
                sweeps = json.loads(_ARCHIVE.read_text())
            for sweep in reversed(sweeps):
                sel = np.asarray(sweep['lab_id']) == c
                a, b = (np.asarray(sweep[k])[sel] for k in ('cmd_stress', 'true_stress'))
                if len(a) >= 2 and np.ptp(a) > 0:
                    result[c] = np.polyfit(a, b, 1)
                    break
            else:
                raise ValueError(f'No usable load-cell calibration for channel {c}')
        else:
            raise ValueError(f'Load-cell commands have no variation for channel {c}')
    return result


def _prepare(dataset):
    def join(key):
        return np.concatenate((dataset['train'][key], dataset['test'][key]))

    lab, stratum = join('lab_id'), join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    for c in np.unique(lab):
        select = lab == c
        panel = ref[select]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        spread = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[select] = (x[select] - center) / spread
    _, cell = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    observed = np.isfinite(x)
    sums = np.zeros((cell.max()+1, x.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(observed, x, 0.))
    np.add.at(counts, cell, observed)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    fill = means.sum(axis=0) / np.count_nonzero(counts, axis=0)
    x = np.where(observed, x, fill)
    static = np.column_stack((np.ones(len(x)), x, np.sign(hold)*np.sqrt(np.abs(hold))))
    return past, hold, static


def _design(tau, state, dt, time):
    past, hold, static = state
    # Unroll the memory recurrence: each commanded step decays over the later steps.
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = -np.expm1(-dt[:, None]/tau) * np.exp(-tail[:, None]/tau)
    gap = hold[:, None] - past @ kernel
    expo = -np.expm1(-time[:, None]/tau)
    phi = time/(1.0 + 0.85*time)
    diff = gap[:, 0]-gap[:, 1]
    curve = np.concatenate((gap[:, None, :]*expo,
                            static[:, None, :]*phi[None, :, None],
                            diff[:, None, None]*expo[None, :, :1]), axis=2)
    rate = np.column_stack((gap/tau, static, diff/tau[0]))
    return curve, rate


def _mad(x, axis=None):
    return 1.4826*np.nanmedian(np.abs(x-np.nanmedian(x, axis=axis, keepdims=True)), axis=axis)


def _fit(state, dt, time, delta, rate):
    cm, rm = np.isfinite(delta), np.isfinite(rate)
    y = np.r_[delta[cm], rate[rm]]
    n_curve = cm.sum()
    def design(p):
        a, b = _design(np.exp(p), state, dt, time)
        return np.concatenate((a[cm], b[rm]))

    # Before estimating instrument noise, keep the initial-rate scale comparable
    # to the observed strain changes near the start of the hold.
    w = np.r_[np.ones(n_curve), np.full(rm.sum(), min(time[time > 0]))]
    bounds = np.log([max(0.02*np.min(dt), 1e-4), 20*(dt.sum()+time.max())])
    starts = np.clip(np.log([[.25, 2.5], [.65, 4.], [1.25, 8.], [.15, 9.]]), *bounds)

    def profile(p, weights):
        a = design(p)
        coef = np.linalg.lstsq(a*weights[:, None], y*weights, rcond=None)[0]
        return a @ coef-y, coef

    def opt(p):
        return least_squares(lambda q: profile(q, w)[0]*w, p, bounds=bounds,
                             diff_step=1e-4, max_nfev=100, ftol=1e-9)

    p = min((opt(s) for s in starts), key=lambda fit: fit.cost).x
    for _ in range(10):
        residual, _ = profile(p, w)
        e = np.full(delta.shape, np.nan)
        e[cm] = residual[:n_curve]
        sigma_c = np.maximum(_mad(e, axis=0), max(0.35*_mad(e), 1e-7))
        sigma_r = max(_mad(residual[n_curve:]), 1e-7) if rm.any() else 1.
        sigma = np.r_[np.broadcast_to(sigma_c, delta.shape)[cm], np.full(rm.sum(), sigma_r)]
        w = 1. / np.hypot(sigma, residual/2.38)
        p = opt(p).x
    return np.exp(p), profile(p, w)[1]


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, time = dataset['past_dt'], dataset['eval_times']
    state = _prepare(dataset)
    n = len(train['lab_id'])
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, coef = _fit(tuple(a[:n] for a in state), dt, time, delta, train['instant_rate_obs'])
    if tau[0] > tau[1]:
        tau = tau[::-1].copy()
        # The asymmetric cross-gap exponential follows the fast branch.
        # Refit in the required fast-to-slow orientation rather than swapping coefficients.
        a, b = _design(tau, tuple(s[:n] for s in state), dt, time)
        cm, rm = np.isfinite(delta), np.isfinite(train['instant_rate_obs'])
        coef = np.linalg.lstsq(np.concatenate((a[cm], b[rm]*time[0])),
                               np.r_[delta[cm], train['instant_rate_obs'][rm]*time[0]], rcond=None)[0]
    a, b = _design(tau, tuple(s[n:] for s in state), dt, time)
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:], 'predicted_strain': test['strain_0'][:, None]+a @ coef,
            'initial_rate_pred': b @ coef}
