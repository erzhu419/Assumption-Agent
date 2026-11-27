"""Calibrated two-branch Prony model for terminal polymer flow."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _archive_calibration(lab):
    try:
        with open(_ARCHIVE) as stream:
            history = json.load(stream)
    except FileNotFoundError:
        return None
    if isinstance(history, dict):
        if 'stress_gains' in history and 'stress_biases' in history:
            return np.array([history['stress_gains'][lab], history['stress_biases'][lab]])
        history = history.get('records', history.get(str(lab), []))
    if isinstance(history, dict):
        history = [history]
    if history and isinstance(history[0], (list, tuple, int, float)):
        return np.asarray(history, dtype=float) if np.shape(history) == (2,) else None
    fits = [[r['gain'], r['bias']] for r in history
            if int(r.get('lab_id', lab)) == lab and 'gain' in r and 'bias' in r]
    return np.median(fits, axis=0) if fits else None


def _calibrate(cal):
    result = []
    for lab in range(4):
        rows = np.asarray(cal['lab_id']) == lab
        x = np.asarray(cal['cmd_stress'])[rows]
        y = np.asarray(cal['true_stress'])[rows]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        archived = _archive_calibration(lab) if len(x) < 2 else None
        if len(x) >= 3 and np.ptp(x) > 0:
            # Screen against a clean leave-one-out fit: the spike can tilt
            # the full fit enough to hide its own residual.
            full_error = np.abs(y - np.polyval(np.polyfit(x, y, 1), x))
            if full_error.max() > 0.01:
                candidates = []
                for j in range(len(x)):
                    keep = np.arange(len(x)) != j
                    if np.ptp(x[keep]) == 0:
                        continue
                    residual = np.abs(y - np.polyval(np.polyfit(x[keep], y[keep], 1), x))
                    if residual[j] > 3.0 and residual[keep].max() <= 0.01:
                        candidates.append(j)
                if candidates:
                    if len(candidates) > 1 or len(x) <= 4:
                        archived = _archive_calibration(lab)
                    spike = candidates[0] if len(candidates) == 1 else None
                    if spike is None and archived is not None:
                        residual = np.abs(y - np.polyval(archived, x))
                        j = int(np.argmax(residual))
                        if j in candidates and residual[j] > 3.0:
                            spike = j
                    if spike is not None:
                        keep = np.arange(len(x)) != spike
                        x, y = x[keep], y[keep]
                else:
                    i, j = np.triu_indices(len(x), 1)
                    dx = x[j] - x[i]
                    use = dx != 0
                    slope = np.median((y[j] - y[i])[use] / dx[use])
                    offset = np.median(y - slope*x)
                    clean = np.abs(y - (slope*x + offset)) <= 3.0
                    if not clean.all():
                        archived = _archive_calibration(lab)
                        x, y = x[clean], y[clean]
        if len(x) >= 4 and np.ptp(x) > 0:
            fit = np.polyfit(x, y, 1)
        elif archived is not None:
            fit = archived
        elif len(x) >= 2 and np.ptp(x) > 0:
            fit = np.polyfit(x, y, 1)
        else:
            fit = (1.0, np.mean(y-x) if len(x) else 0.0)
        result.append(fit)
    return np.asarray(result)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    def joined(key):
        return np.concatenate((train[key], test[key]), axis=0)
    lab, stratum = joined('lab_id'), joined('stratum')
    ref = joined('ref_coupons')
    aligned = joined('coupon_features_raw').astype(float)
    for c in np.unique(lab):
        rows = lab == c
        panel = ref[rows]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (aligned[rows] - center)/scale
    cells = np.column_stack((lab, stratum))
    means = []
    for cell in np.unique(cells, axis=0):
        values = aligned[np.all(cells == cell, axis=1)]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(aligned.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isnan(aligned), np.nanmean(means, axis=0), aligned)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = joined('past_stress_cmd') * gain[:, None] + bias[:, None]
    hold = joined('stress_hold_cmd') * gain + bias
    baseline = joined('strain_0')
    return past, hold, np.column_stack((np.ones(len(lab)), aligned)), baseline


def _design(taus, past, hold, features, dt, t):
    # Each history sample contributes its exact final-time memory weight.
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = past @ (np.exp(-trailing[:, None]/taus) * -np.expm1(-dt[:, None]/taus))
    gap = hold[:, None] - memory
    prony = gap[:, None, :] * -np.expm1(-t[None, :, None]/taus)
    flow = features[:, None, :] * t[None, :, None]
    return np.concatenate((prony, flow), axis=2), np.column_stack((gap/taus, features))


def _mad(x):
    return max(1e-6, 1.4826*np.median(np.abs(x-np.median(x))))


def _fit(past, hold, features, dt, t, y, rates):
    ok = np.isfinite(y)
    rok = np.isfinite(rates)
    target = np.r_[y[ok], rates[rok]]
    count = ok.sum()
    def matrix(logtau):
        a, b = _design(np.exp(logtau), past, hold, features, dt, t)
        return np.concatenate((a[ok], b[rok]), axis=0)
    def noise(res):
        s1 = _mad(res[:count]); s2 = _mad(res[count:]) if rok.any() else s1
        return np.r_[np.full(count, s1), np.full(rok.sum(), s2)]
    tmin = min(t[t > 0].min(), dt[dt > 0].min())/20
    tmax = max(t.max(), dt.sum())*20
    bounds = (np.r_[np.repeat(np.log(tmin), 2), np.full(7, -np.inf)],
              np.r_[np.repeat(np.log(tmax), 2), np.full(7, np.inf)])
    def optimize(start, scale, loss):
        def residual(p):
            return (matrix(p[:2]) @ p[2:] - target)/scale
        return least_squares(residual, start, bounds=bounds, loss=loss, x_scale='jac',
                             max_nfev=200)
    seeds = t.max()*np.array([[.05,.28], [.08,.65], [.13,1.6], [.24,.65], [.025,2.]])
    starts = []
    for tau in seeds:
        logtau = np.log(tau)
        mat = matrix(logtau)
        w = np.linalg.lstsq(mat, target, rcond=None)[0]
        starts.append(np.r_[logtau, w])
    scale = noise(matrix(starts[0][:2]) @ starts[0][2:] - target)
    pilot = min((optimize(p, scale, 'soft_l1') for p in starts), key=lambda fit: fit.cost)
    scale = 2.0*noise(pilot.fun*scale)
    fit = min((optimize(p, scale, 'cauchy') for p in [pilot.x, *starts]), key=lambda fit: fit.cost)
    scale = 2.0*noise(fit.fun*scale)
    p = optimize(fit.x, scale, 'cauchy').x
    order = np.argsort(p[:2])
    return np.exp(p[:2][order]), np.r_[p[2:4][order], p[4:]]


def solve_alpha_m97_c03_s00(dataset: dict) -> dict:
    past, hold, features, baseline = _prepare(dataset)
    t = np.asarray(dataset['eval_times'])
    dt = np.asarray(dataset['past_dt'])
    train = dataset['train']
    n = len(train['lab_id'])
    taus, weights = _fit(past[:n], hold[:n], features[:n], dt, t,
                         train['strain_curves_obs'] - baseline[:n, None],
                         np.asarray(train['instant_rate_obs']))
    a, b = _design(taus, past[n:], hold[n:], features[n:], dt, t)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'regime_mode_weights': np.zeros(2),
            'predicted_strain': baseline[n:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
