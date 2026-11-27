"""Calibrated two-branch creep with linear terminal flow."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _calibrate(rows):
    result = np.empty((4, 2))
    sweeps = None
    cmd, true = (np.asarray(rows[k], dtype=float) for k in ('cmd_stress', 'true_stress'))
    for lab in range(4):
        sel = (np.asarray(rows['lab_id']) == lab) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[sel], true[sel]
        spikes = False
        if len(x) >= 3 and np.ptp(x) > 0:
            dx, dy = x[:, None]-x, y[:, None]-y
            slopes = [np.median(b[a != 0]/a[a != 0]) for a, b in zip(dx, dy) if np.any(a != 0)]
            gain = np.median(slopes)
            bias = np.median(y-gain*x)
            clean = np.abs(y-gain*x-bias) <= 3.0
            spikes = not clean.all()
            x, y = x[clean], y[clean]
        usable = len(x) >= 2 and np.ptp(x) > 0
        if usable and (len(x) >= 4 or not spikes):
            result[lab] = np.polyfit(x, y, 1)
            continue
        if len(x) < 2 or spikes:
            if sweeps is None:
                sweeps = json.loads(_ARCHIVE.read_text()) if _ARCHIVE.is_file() else []
            for sweep in reversed(sweeps):
                mask = np.asarray(sweep['lab_id']) == lab
                a, b = (np.asarray(sweep[k])[mask] for k in ('cmd_stress', 'true_stress'))
                if len(a) >= 2 and np.ptp(a) > 0:
                    result[lab] = np.polyfit(a, b, 1)
                    break
            else:
                result[lab] = np.polyfit(x, y, 1) if usable else (1., np.median(y-x) if len(x) else 0.)
        else:
            result[lab] = (1., np.median(y-x))
    return result


def _prepare(dataset):
    def join(k):
        return np.concatenate((dataset['train'][k], dataset['test'][k]))

    lab, stratum = join('lab_id'), join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    x, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    for channel in np.unique(lab):
        sel = lab == channel
        panel = ref[sel]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        x[sel] = (x[sel]-np.median(np.nanmean(panel, axis=0)))/np.median(np.nanstd(panel, axis=0, ddof=0))
    _, cell = np.unique(np.column_stack((lab, stratum)), axis=0, return_inverse=True)
    observed = np.isfinite(x)
    sums = np.zeros((cell.max()+1, x.shape[1]))
    counts = np.zeros_like(sums)
    np.add.at(sums, cell, np.where(observed, x, 0.))
    np.add.at(counts, cell, observed)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    n_cells = np.count_nonzero(counts, axis=0)
    fill = np.divide(means.sum(axis=0), n_cells, out=np.zeros(x.shape[1]), where=n_cells > 0)
    x = np.where(observed, x, fill)
    return past, hold, np.column_stack((np.ones(len(x)), x))


def _design(logtau, state, dt, times):
    past, hold, x = state
    tau = np.exp(logtau)
    z = np.zeros((len(hold), 2))
    for width, stress in zip(dt, past.T):
        u = -width/tau
        z = np.exp(u)*z - np.expm1(u)*stress[:, None]
    gap = hold[:, None]-z
    curves = np.concatenate((gap[:, None, :]*(-np.expm1(-times[:, None]/tau)),
                             x[:, None, :]*times[None, :, None]), axis=2)
    rates = np.column_stack((gap/tau, x))
    return curves, rates


def _scale(e):
    return max(1.4826*np.median(np.abs(e-np.median(e))), 1e-7)


def _fit(state, dt, times, delta, rate_obs, loss='cauchy'):
    cm, rm = np.isfinite(delta), np.isfinite(rate_obs)
    y = delta[cm]
    r = rate_obs[rm]
    horizon = np.broadcast_to(times, delta.shape)[cm]
    # Exclude gross values from initialization only; the final loss handles them directly.
    center = np.median(y/horizon)
    initial = np.abs(y/horizon-center) < 10*_scale(y/horizon)
    bounds = np.log([max(0.02*np.min(dt), 1e-4), 20*(dt.sum()+times.max())])
    starts = np.clip(np.log([[.25, 2.5], [.65, 4.], [1.25, 8.]]), *bounds)

    def matrices(p):
        a, b = _design(p, state, dt, times)
        return a[cm], b[rm]

    def mul(a, w):
        return np.einsum('ij,j->i', a, w)

    def warm_start(p):
        a, _ = matrices(p)
        w = np.linalg.lstsq(a[initial], y[initial], rcond=None)[0]
        return mul(a[initial], w)-y[initial]

    warm = min((least_squares(warm_start, p, bounds=bounds, max_nfev=70) for p in starts), key=lambda f: f.cost)
    a, b = matrices(warm.x)
    w = np.linalg.lstsq(a[initial], y[initial], rcond=None)[0]
    cs = _scale(mul(a, w)-y)
    rs = _scale(mul(b, w)-r) if len(r) else 1.

    def profile(p, return_weights=False):
        a, b = matrices(p)
        a = np.vstack((a/cs, b/rs))
        yy = np.r_[y/cs, r/rs]
        w = np.linalg.lstsq(a, yy, rcond=None)[0]
        if loss != 'linear':
            for _ in range(18):
                e = (mul(a, w)-yy)/2.5
                v = 1/np.hypot(1., e)
                if loss == 'soft_l1':
                    v = np.sqrt(v)
                w_next = np.linalg.lstsq(a*v[:, None], yy*v, rcond=None)[0]
                if np.linalg.norm(w_next-w) < 1e-8*(1+np.linalg.norm(w)):
                    w = w_next
                    break
                w = w_next
        if return_weights:
            return w
        e = mul(a, w)-yy
        if loss == 'cauchy':
            return np.sign(e)*2.5*np.sqrt(np.log1p((e/2.5)**2))
        if loss == 'soft_l1':
            return e*np.sqrt(2/(np.hypot(1., e/2.5)+1))
        return e

    best = min((least_squares(profile, p, bounds=bounds, max_nfev=75) for p in (warm.x, starts[1])), key=lambda f: f.cost)
    p, w = best.x.copy(), profile(best.x, True)
    if p[0] > p[1]:
        p = p[::-1]
        w[:2] = w[1::-1]
    return p, w


def solve_alpha_m97_c01_batch_02(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    n = len(train['lab_id'])
    state = _prepare(dataset)
    delta = train['strain_curves_obs'] - train['strain_0'][:, None]
    p, w = _fit(tuple(a[:n] for a in state), dt, times, delta, train['instant_rate_obs'])
    curves, rates = _design(p, tuple(a[n:] for a in state), dt, times)
    return {'estimated_taus': np.exp(p), 'hereditary_weights': w,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': test['strain_0'][:, None] + curves @ w,
            'initial_rate_pred': rates @ w}
