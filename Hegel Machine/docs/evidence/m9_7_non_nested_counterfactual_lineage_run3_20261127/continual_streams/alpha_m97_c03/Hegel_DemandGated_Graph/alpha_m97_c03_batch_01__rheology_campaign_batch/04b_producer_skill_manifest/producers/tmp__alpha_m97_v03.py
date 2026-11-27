"""Calibrated two-branch Prony model for the crystallization campaign."""
import json

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = '/tmp/hegel_continual_archive/archive_calibration_history.json'


def _archive_calibration(lab):
    with open(_ARCHIVE) as stream:
        history = json.load(stream)
    fits = []
    for sweep in history:
        rows = np.asarray(sweep['lab_id']) == lab
        x = np.asarray(sweep['cmd_stress'], dtype=float)[rows]
        y = np.asarray(sweep['true_stress'], dtype=float)[rows]
        ok = np.isfinite(x) & np.isfinite(y)
        x, y = x[ok], y[ok]
        if x.size >= 2 and np.ptp(x) > 0:
            fits.append(np.polyfit(x, y, 1))
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


def _design(taus, past, hold, features, dt, t, damage, plateau):
    trailing = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = past @ (np.exp(-trailing[:, None] / taus) * -np.expm1(-dt[:, None] / taus))
    gap = hold[:, None] - memory
    relax = -np.expm1(-t[:, None] / taus)
    act = np.tanh(damage)
    clock = np.log1p(t)[None, :] if plateau else t[None, :]
    extra = act * hold
    offset = act * memory[:, 0]
    curves = np.concatenate((gap[:, None, :] * relax,
                             features[:, None, :] * clock[:, :, None],
                             (extra[:, None] * relax[:, 0])[:, :, None],
                             (offset[:, None] * clock)[:, :, None]), axis=2)
    rates = np.column_stack((gap / taus, features, extra / taus[0], offset))
    return curves, rates


def _fit(past, hold, features, dt, t, damage, y, rates, plateau):
    valid_y, valid_r = np.isfinite(y), np.isfinite(rates)
    target = np.r_[y[valid_y], rates[valid_r]]
    n_curve = valid_y.sum()
    n_rate = valid_r.sum()
    k = 9 if np.any(damage > 0) else 7

    def matrix(log_tau):
        a, b = _design(np.exp(log_tau), past, hold, features, dt, t, damage, plateau)
        return np.vstack((a[valid_y, :k], b[valid_r, :k]))

    def scales(residual):
        def mad(x):
            return max(1e-6, 1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else 1.
        return np.repeat([mad(residual[:n_curve]), mad(residual[n_curve:])], [n_curve, n_rate])

    lo = np.log(min(t[t > 0].min(), dt[dt > 0].min()) / 20)
    hi = np.log(max(t.max(), dt.sum()) * 20)
    bounds = (np.r_[lo, lo, np.full(k, -np.inf)], np.r_[hi, hi, np.full(k, np.inf)])

    def raw_residual(p):
        return np.einsum('ij,j->i', matrix(p[:2]), p[2:]) - target

    def optimize(p, scale, loss):
        return least_squares(lambda q: raw_residual(q) / scale, p, bounds=bounds,
                             loss=loss, x_scale='jac', tr_solver='lsmr', max_nfev=130)

    starts = []
    for seed in ((.09, .48), (.045, .95), (.19, .7), (.13, 1.65), (.48, .09)):
        lt = np.log(t.max() * np.asarray(seed)).clip(lo + 1e-8, hi - 1e-8)
        starts.append(np.r_[lt, np.linalg.lstsq(matrix(lt), target, rcond=None)[0]])
    scale = scales(raw_residual(starts[0]))
    pilot = min((optimize(p, scale, 'soft_l1') for p in starts), key=lambda f: f.cost)
    scale = 2 * scales(raw_residual(pilot.x))
    fit = min((optimize(p, scale, 'cauchy') for p in [pilot.x, *starts]), key=lambda f: f.cost)
    fit = optimize(fit.x, 2 * scales(raw_residual(fit.x)), 'cauchy')
    taus, weights = np.exp(fit.x[:2]), np.pad(fit.x[2:], (0, 9-k))
    if k == 7 and taus[0] > taus[1]:
        taus = taus[::-1]
        weights[:2] = weights[:2][::-1]
    return taus, weights


def solve_alpha_m97_c03_batch_01(dataset: dict) -> dict:
    past, hold, features, baseline = _prepare(dataset)
    dt, t = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    damage = np.maximum(0, np.abs(past) - dataset['crit_stress'])**2 @ dt
    plateau = dataset['shear_banding_index'] <= .50
    if not plateau:
        damage[:] = 0
    train = dataset['train']
    n = len(train['lab_id'])
    taus, weights = _fit(past[:n], hold[:n], features[:n], dt, t, damage[:n],
                         train['strain_curves_obs'] - baseline[:n, None],
                         np.asarray(train['instant_rate_obs']), plateau)
    a, b = _design(taus, past[n:], hold[n:], features[n:], dt, t, damage[n:], plateau)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': baseline[n:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
