"""Calibrated two-branch Prony model for strain-crystallized flow."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _historical_line(records, channel):
    if isinstance(records, list):
        for record in reversed(records):
            line = _historical_line(record, channel)
            if line is not None:
                return line
    elif isinstance(records, dict):
        if {'lab_id', 'cmd_stress', 'true_stress'} <= records.keys():
            lab, cmd, true = (np.asarray(records[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
            use = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
            x, y = cmd[use], true[use]
            clean = _clean_sweep(x, y)
            if np.unique(x[clean]).size >= 2:
                return _line(x[clean], y[clean])
    return None


def _clean_sweep(x, y):
    if np.unique(x).size < 2:
        return np.ones(x.size, dtype=bool)
    # A spike can tilt a short sweep enough to hide in its fitted residuals.
    # Test it against the line supported by the other readings instead.
    if x.size >= 3:
        candidates = []
        for omit in range(x.size):
            keep = np.arange(x.size) != omit
            if np.unique(x[keep]).size < 2:
                continue
            gain, bias = _line(x[keep], y[keep])
            residual = y - (gain * x + bias)
            # Clean public and prefix sweeps agree with an affine line well
            # within 0.01 stress units; require that agreement before removal.
            if np.max(np.abs(residual[keep])) < 0.01 and abs(residual[omit]) > 3.0:
                candidates.append((np.sum(residual[keep]**2), keep))
        if candidates:
            return min(candidates, key=lambda candidate: candidate[0])[1]
    i, j = np.triu_indices(x.size, 1)
    distinct = x[j] != x[i]
    i, j = i[distinct], j[distinct]
    slope = np.median((y[j] - y[i]) / (x[j] - x[i]))
    return np.abs(y - slope * x - np.median(y - slope * x)) <= 3.0


def _line(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(x.size))), y, rcond=None)[0]


def _calibrate(cal):
    lines = np.empty((4, 2))
    history = None
    lab, cmd, true = (np.asarray(cal[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    for c in range(4):
        take = (lab == c) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        clean = _clean_sweep(x, y)
        if np.unique(x[clean]).size >= 2 and (clean.sum() >= 4 or clean.all()):
            lines[c] = _line(x[clean], y[clean])
        else:
            if history is None:
                history = json.loads(_HISTORY.read_text())
            line = _historical_line(history, c)
            if line is None:
                raise ValueError(f'No usable calibration for channel {c}')
            lines[c] = line
    return lines


def _align(train, test):
    labs, strata, raw, panel = (np.concatenate((train[k], test[k])) for k in
                                ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons'))
    aligned = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        rows = labs == lab
        reference = panel[rows]
        reference = reference[:, np.isfinite(reference).any(axis=0)]
        center = np.median(np.nanmean(reference, axis=0))
        scale = np.median(np.nanstd(reference, axis=0, ddof=0))
        aligned[rows] = (raw[rows] - center) / scale
    cells, ids = np.unique(np.column_stack((labs, strata)), axis=0, return_inverse=True)
    means = np.empty((len(cells), raw.shape[1]))
    for i in range(len(cells)):
        values = aligned[ids == i]
        count = np.isfinite(values).sum(axis=0)
        means[i] = np.divide(np.nansum(values, axis=0), count,
                             out=np.full(raw.shape[1], np.nan), where=count > 0)
    fills = means.copy()
    for i, j in zip(*np.where(~np.isfinite(means))):
        available = np.isfinite(means[:, j])
        same_lab = available & (cells[:, 0] == cells[i, 0])
        use = same_lab if same_lab.any() else available
        fills[i, j] = means[use, j].mean() if use.any() else 0.0
    completed = np.where(np.isfinite(aligned), aligned, fills[ids])
    n_train = len(train['lab_id'])
    return completed[:n_train], completed[n_train:]


def _inputs(part, aligned, lines):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned))


def _design(logtau, inputs, dt, times, crit, crystallizes):
    stress, hold, xd = inputs
    tau = np.exp(logtau)
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    kernel = np.exp(-tail[:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    z = stress @ kernel
    gap = hold[:, None] - z
    damage = np.maximum(np.abs(stress) - crit, 0.0)**2 @ dt
    act = np.tanh(damage) if crystallizes else np.zeros_like(damage)
    # Low banding sets the base drift law; damage gates the two corrections.
    flow = np.broadcast_to(np.log1p(times) if crystallizes else times,
                           (len(hold), len(times)))
    decay = -np.expm1(-times[:, None] / tau)
    a = np.empty((len(hold), len(times) + 1, 9))
    a[:, :-1, :2] = gap[:, None, :] * decay
    a[:, -1, :2] = gap / tau
    a[:, :-1, 2:7] = xd[:, None, :] * flow[:, :, None]
    a[:, -1, 2:7] = xd
    a[:, :-1, 7] = (act * hold)[:, None] * decay[:, 0]
    a[:, -1, 7] = act * hold / tau[0]
    a[:, :-1, 8] = (act * z[:, 0])[:, None] * flow
    a[:, -1, 8] = act * z[:, 0]
    return a


def _fit(inputs, dt, times, crit, crystallizes, y):
    # Profile all nine linear coefficients at each pair of relaxation times.
    valid = np.isfinite(y)
    target = y[valid]
    lo = min(np.min(dt), np.min(times[times > 0])) / 10
    hi = max(np.sum(dt), np.max(times)) * 10

    sqrt_weight = np.ones(target.shape)

    def profile(logtau, return_weights=False):
        a = _design(logtau, inputs, dt, times, crit, crystallizes)[valid]
        aw = a * sqrt_weight[:, None]
        weights = np.linalg.lstsq(aw, target * sqrt_weight, rcond=1e-11)[0]
        return weights if return_weights else (a @ weights - target) * sqrt_weight

    def optimize(start):
        return least_squares(profile, start, bounds=(np.log(lo), np.log(hi)),
                             max_nfev=100, ftol=1e-10, gtol=1e-9, xtol=1e-9)

    starts = ((0.45, 2.5), (1.2, 6.0), (0.22, 1.3), (3.0, 0.65))
    fits = [optimize(np.log(np.clip(seed, lo * 1.01, hi / 1.01))) for seed in starts]
    logtau = min(fits, key=lambda fit: fit.cost).x
    # Isolated readout spikes affect individual cells, not entire specimens.
    # Estimate a separate residual scale for each curve horizon and the rate.
    for _ in range(12):
        resid = _design(logtau, inputs, dt, times, crit, crystallizes) @ profile(logtau, True) - y
        mad = np.nanmedian(np.abs(resid - np.nanmedian(resid, axis=0)), axis=0)
        scale = np.maximum(2.385 * 1.4826 * mad, 1e-6)
        sqrt_weight = (1 / np.hypot(1, resid / scale))[valid]
        logtau = optimize(logtau).x
    if not crystallizes:
        logtau = np.sort(logtau)
    return logtau, profile(logtau, return_weights=True)


def solve_alpha_m97_c03_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    lines = _calibrate(dataset['calibration_load_cells'])
    x_train, x_test = _align(train, test)
    train_inputs = _inputs(train, x_train, lines)
    test_inputs = _inputs(test, x_test, lines)
    times = np.asarray(dataset['eval_times'], dtype=float)
    dt = np.asarray(dataset['past_dt'], dtype=float)
    crit = float(dataset['crit_stress'])
    crystallizes = dataset['shear_banding_index'] <= 0.50
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    logtau, weights = _fit(train_inputs, dt, times, crit, crystallizes, y)
    pred = _design(logtau, test_inputs, dt, times, crit, crystallizes) @ weights
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': pred[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
