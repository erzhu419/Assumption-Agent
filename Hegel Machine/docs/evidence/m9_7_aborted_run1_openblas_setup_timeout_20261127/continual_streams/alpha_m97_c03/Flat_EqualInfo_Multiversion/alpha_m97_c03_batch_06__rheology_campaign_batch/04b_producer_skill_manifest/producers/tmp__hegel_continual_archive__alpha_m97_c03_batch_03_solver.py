"""Calibrated two-branch Prony model for shear-banding jammed slip."""
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
    return np.where(np.isfinite(aligned), aligned, fills[ids])


def _inputs(part, aligned, lines):
    gain, bias = lines[part['lab_id']].T
    stress = part['past_stress_cmd'] * gain[:, None] + bias[:, None]
    hold = part['stress_hold_cmd'] * gain + bias
    return stress, hold, np.column_stack((np.ones(len(hold)), aligned))


def _design(logtau, inputs, dt, times):
    stress, hold, xd = inputs
    tau = np.exp(logtau)
    remaining = np.r_[np.cumsum(dt[::-1])[::-1], 0.0]
    # Each stress step contributes its own relaxation increment, decayed
    # through all subsequent steps (zero memory before the first step).
    kernel = np.exp(-remaining[1:, None] / tau) * -np.expm1(-dt[:, None] / tau)
    gap = hold[:, None] - stress @ kernel
    diff = gap[:, 0] - gap[:, 1]
    relaxation = -np.expm1(-times[:, None] / tau)
    flow = times / (1.0 + 0.85 * times)
    slip = np.sign(hold) * np.sqrt(np.abs(hold))
    a = np.empty((len(hold), len(times) + 1, 9))
    a[:, :-1, :2] = gap[:, None, :] * relaxation
    a[:, -1, :2] = gap / tau
    a[:, :-1, 2:7] = xd[:, None, :] * flow[None, :, None]
    a[:, -1, 2:7] = xd
    a[:, :-1, 7] = slip[:, None] * flow
    a[:, -1, 7] = slip
    a[:, :-1, 8] = diff[:, None] * relaxation[:, 0]
    a[:, -1, 8] = diff / tau[0]
    return a


def _fit(inputs, dt, times, y):
    lo = np.log(max(min(np.min(dt), np.min(times[times > 0])) / 20, 1e-5))
    hi = np.log(max(np.sum(dt), np.max(times)) * 20)
    bounds = (np.r_[lo, lo, np.full(9, -np.inf)],
              np.r_[hi, hi, np.full(9, np.inf)])
    valid = np.isfinite(y)
    yobs = y[valid]

    # Interpolation in the terminal-flow coordinate removes coupon drift.
    # The resulting local differences are a robust initial noise estimate.
    flow = times / (1 + 0.85 * times)
    alpha = (flow[1:-1] - flow[:-2]) / (flow[2:] - flow[:-2])
    curve = y[:, :-1]
    local = (curve[:, 1:-1] - (1-alpha) * curve[:, :-2] - alpha * curve[:, 2:])
    local /= np.sqrt(1 + (1-alpha)**2 + alpha**2)
    local -= np.nanmedian(local, axis=0)
    width = max(1.4826 * np.nanmedian(np.abs(local)), 1e-7)
    widths = np.full(y.shape[1], width)

    def run(p, loss, multiplier):
        scale = np.broadcast_to(widths, y.shape)[valid]

        def residual(p):
            a = _design(p[:2], inputs, dt, times)[valid]
            return (np.einsum('ij,j->i', a, p[2:]) - yobs) / scale

        return least_squares(residual, p, bounds=bounds, loss=loss,
                             f_scale=multiplier, x_scale='jac', max_nfev=150)

    fastest = min(np.median(dt), np.min(times[times > 0]))
    horizon = np.max(times)
    trials = []
    for start in ((fastest, horizon / 2), (2*fastest, horizon),
                  (fastest / 2, 2*horizon), (horizon / 4, horizon)):
        lt = np.log(start)
        a = _design(lt, inputs, dt, times)
        rate = valid[:, -1]
        beta = np.linalg.lstsq(a[rate, -1], y[rate, -1], rcond=None)[0]
        warm = run(np.r_[lt, beta], 'soft_l1', 6.)
        trial = run(warm.x, 'cauchy', 2.5)
        if trial.x[0] < trial.x[1]:
            trials.append(trial)
    if not trials:
        raise ValueError('Could not fit ordered relaxation times')
    p = min(trials, key=lambda result: result.cost).x
    res = _design(p[:2], inputs, dt, times) @ p[2:] - y
    mad = 1.4826 * np.nanmedian(np.abs(res - np.nanmedian(res, axis=0)), axis=0)
    widths[:-1] = max(np.nanmedian(mad[:-1]), width * 0.35)
    widths[-1] = max(mad[-1], width * 0.35) if np.isfinite(mad[-1]) else width
    refined = run(p, 'cauchy', 2.5).x
    if refined[0] < refined[1]:
        p = refined
    return p[:2], p[2:]


def solve_alpha_m97_c03_batch_03(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    dt, times = dataset['past_dt'], dataset['eval_times']
    lines = _calibrate(dataset['calibration_load_cells'])
    aligned = _align(train, test)
    inputs = _inputs(train, aligned[:n], lines)
    y = np.column_stack((train['strain_curves_obs'] - train['strain_0'][:, None],
                         train['instant_rate_obs']))
    # Missing measured descriptors introduce a coupon-specific flow error.
    # Estimate the shared material law on the directly observed coupons.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    logtau, beta = _fit(tuple(a[complete] for a in inputs), dt, times, y[complete])
    design = _design(logtau, _inputs(test, aligned[n:], lines), dt, times)
    pred = np.einsum('nkj,j->nk', design, beta)
    return {'estimated_taus': np.exp(logtau),
            'hereditary_weights': beta[:7],
            'regime_mode_weights': beta[7:],
            'predicted_strain': pred[:, :-1] + test['strain_0'][:, None],
            'initial_rate_pred': pred[:, -1]}
