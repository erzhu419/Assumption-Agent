"""Calibrate load cells and fit a two-branch crystallization plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _channel(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    mask = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def _line(x, y, strict=False):
    if len(x) < 2 or np.ptp(x) == 0:
        return None
    design = np.column_stack((x, np.ones(len(x))))
    coeff = np.linalg.lstsq(design, y, rcond=None)[0]
    # Archived sweeps resolve a static calibration to much better than 0.02.
    # Use that redundancy to reject sparse sweeps made entirely of EMI points.
    tol = 0.02 if strict else 3.0
    if np.max(np.abs(y - design @ coeff)) <= tol:
        return coeff
    best = None
    for i, j in zip(*np.triu_indices(len(x), k=1)):
        if x[i] == x[j]:
            continue
        gain = (y[j] - y[i]) / (x[j] - x[i])
        error = np.abs(y - y[i] - gain * (x - x[i]))
        keep = error <= tol
        count = np.count_nonzero(keep)
        key = (count, -np.dot(error[keep], error[keep]))
        if best is None or key > best[0]:
            best = key, keep
    if best is not None and best[0][0] >= 3:
        keep = best[1]
        return np.linalg.lstsq(design[keep], y[keep], rcond=None)[0]
    return None


def _calibrate(sweep):
    coeffs = []
    history = None
    for lab in range(4):
        x, y = _channel(sweep, lab)
        coeff = _line(x, y, strict=True)
        if coeff is None:
            # With two spikes in three rows, an in-batch fit can absorb both.
            # A prior line can identify that pattern if a clean point anchors it.
            if history is None:
                history = json.loads(_HISTORY.read_text())
            direct = _line(x, y)
            for past_sweep in reversed(history):
                px, py = _channel(past_sweep, lab)
                prior = _line(px, py, strict=True)
                if prior is None:
                    continue
                if direct is None:
                    direct = prior
                    break
                residual = np.abs(y - (prior[0] * x + prior[1]))
                clean = residual <= 0.02
                if len(px) >= 4 and clean.any() and (residual > 3.0).any():
                    direct = _line(x[clean], y[clean], strict=True)
                    if direct is None:
                        direct = prior
                    break
            if direct is None:
                raise ValueError(f'No clean calibration for load cell {lab}')
            coeff = direct
        coeffs.append(coeff)
    return np.asarray(coeffs)


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, stratum = join('lab_id'), join('stratum')
    refs, raw = join('ref_coupons'), join('coupon_features_raw')
    aligned = np.empty_like(raw, dtype=float)
    cells = []
    for channel in np.unique(lab):
        rows = lab == channel
        panel = refs[rows]
        center = np.nanmedian(np.nanmean(panel, axis=0))
        scale = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - center) / scale
        for s in np.unique(stratum[rows]):
            values = aligned[rows & (stratum == s)]
            count = np.isfinite(values).sum(axis=0)
            cells.append(np.divide(np.nansum(values, axis=0), count,
                                   out=np.full(raw.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(cells, axis=0))
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None] * join('past_stress_cmd') + bias[:, None]
    hold = gain * join('stress_hold_cmd') + bias
    dt = np.asarray(dataset['past_dt'])
    damage = np.maximum(0, np.abs(past) - dataset['crit_stress'])**2 @ dt
    # The low-shear network sets the time kernel; damage gates only the
    # two additive regime terms, which vanish continuously at zero damage.
    low_shear = dataset['shear_banding_index'] <= 0.50
    act = np.tanh(damage) * low_shear
    return past, hold, np.column_stack((np.ones(len(hold)), aligned)), act, low_shear


def _design(past, hold, drift, act, low_shear, dt, times, tau):
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    memory = past @ (-np.expm1(-dt[:, None] / tau) * np.exp(-remaining[:, None] / tau))
    gap = hold[:, None] - memory
    relax = -np.expm1(-times[:, None] / tau)
    phi = (np.log1p(times) if low_shear else times)[None, :]
    transient, terminal = act * hold, act * memory[:, 0]
    curves = np.concatenate((gap[:, None, :] * relax,
                             drift[:, None, :] * phi[:, :, None],
                             (transient[:, None] * relax[:, 0])[:, :, None],
                             (terminal[:, None] * phi)[:, :, None]), axis=2)
    rates = np.column_stack((gap / tau, drift, transient / tau[0], terminal))
    return curves, rates


def _fit(inputs, dt, times, curves, rates):
    nc = curves.size
    target = np.r_[curves.ravel(), rates]
    measured = np.isfinite(target)
    target = target[measured]
    rate_rows = (np.arange(nc + len(rates)) >= nc)[measured]

    def matrix(logtau):
        a, b = _design(*inputs, dt, times, np.exp(logtau))
        return np.concatenate((a.reshape(-1, 9), b))[measured]

    def profile(logtau):
        a = matrix(logtau)
        return a @ np.linalg.lstsq(a, target, rcond=None)[0] - target

    lo, hi = np.log([0.015, 80.0])
    starts = ([0.25, 2.5], [0.7, 5.0], [2.5, 0.25], [5.0, 0.7])
    fits = [least_squares(profile, np.log(start), bounds=(lo, hi), max_nfev=80)
            for start in starts]

    def scales(residual):
        scale = np.empty_like(residual)
        for rows in (rate_rows, ~rate_rows):
            if rows.any():
                block = residual[rows]
                scale[rows] = max(1e-9, 1.4826 * np.median(np.abs(block - np.median(block))))
        return scale

    scale = scales(min(fits, key=lambda fit: fit.cost).fun)
    bounds = (np.r_[lo, lo, np.full(9, -np.inf)], np.r_[hi, hi, np.full(9, np.inf)])

    def refine(start, scale):
        def residual(params):
            return (matrix(params[:2]) @ params[2:] - target) / scale
        return least_squares(residual, start, bounds=bounds, loss='cauchy',
                             f_scale=2, x_scale='jac', max_nfev=90)

    candidates = []
    for logtau in (fit.x for fit in fits):
        a = matrix(logtau)
        weights = np.linalg.lstsq(a / scale[:, None], target / scale, rcond=None)[0]
        candidates.append(refine(np.r_[logtau, weights], scale))
    best = min(candidates, key=lambda fit: fit.cost)
    best = refine(best.x, scales(best.fun * scale))
    return np.exp(best.x[:2]), best.x[2:]


def solve_alpha_m97_c02_batch_07(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    inputs = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    curves = np.asarray(train['strain_curves_obs']) - np.asarray(train['strain_0'])[:, None]
    tau, weights = _fit((*[column[:n] for column in inputs[:-1]], inputs[-1]), dt, times, curves,
                        np.asarray(train['instant_rate_obs']))
    a, b = _design(*(column[n:] for column in inputs[:-1]), inputs[-1], dt, times, tau)
    return {'estimated_taus': tau, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': np.asarray(test['strain_0'])[:, None] + a @ weights,
            'initial_rate_pred': b @ weights}
