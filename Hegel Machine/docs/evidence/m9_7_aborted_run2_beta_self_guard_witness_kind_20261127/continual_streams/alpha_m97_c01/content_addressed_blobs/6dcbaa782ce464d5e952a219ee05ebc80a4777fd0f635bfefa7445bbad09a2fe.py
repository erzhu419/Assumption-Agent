"""Calibrated two-branch creep with a damage-gated logarithmic plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _line_candidates(x, y):
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    slopes = (y[j] - y[i]) / (x[j] - x[i])
    errors = np.abs(y[i, None] + slopes[:, None] * (x - x[i, None]) - y)
    return errors


def _clean_line(x, y):
    if len(x) < 3:
        return np.ones(len(x), dtype=bool), False
    errors = _line_candidates(x, y)
    if not len(errors):
        return np.ones(len(x), dtype=bool), False
    median = np.median(errors, axis=1)
    best = np.lexsort((np.sum(np.minimum(errors, 3.0), axis=1), median))[0]
    core = errors[best] <= max(8 * median[best], 1e-10)
    gain, bias = _affine(x[core], y[core])
    clean = np.abs(y - gain * x - bias) <= 3.0
    return clean, not clean.all()


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    lab, cmd, true = (np.asarray(rows[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    coefficients = np.empty((4, 2))
    history = None

    def prior(channel):
        nonlocal history
        if history is None:
            history = json.loads(_HISTORY.read_text())
        for sweep in reversed(history):
            take = np.asarray(sweep['lab_id']) == channel
            x, y = (np.asarray(sweep[k])[take] for k in ('cmd_stress', 'true_stress'))
            clean, _ = _clean_line(x, y)
            if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
                return _affine(x[clean], y[clean])
        raise ValueError(f'No usable calibration sweep for lab {channel}')

    for channel in range(4):
        take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        clean, emi = _clean_line(x, y)
        if len(x) == 3:
            # Three rows cannot identify which pair is clean after a spike.
            errors = _line_candidates(x, y)
            if errors.size and errors.max() > 3.0:
                gain, _ = prior(channel)
                intercepts = y - gain * x
                candidate = np.abs(intercepts - np.median(intercepts)) <= 3.0
                if not candidate.all():
                    clean, emi = candidate, True
        if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
            coefficients[channel] = _affine(x[clean], y[clean])
        elif len(x) < 2 or emi:
            gain, bias = prior(channel)
            if clean.any():
                bias = np.median(y[clean] - gain * x[clean])
            coefficients[channel] = gain, bias
        else:
            raise ValueError(f'Calibration commands do not vary for lab {channel}')
    return coefficients


def _inputs(dataset):
    def stack(key):
        return np.concatenate((dataset['train'][key], dataset['test'][key]))

    lab, stratum = stack('lab_id'), stack('stratum')
    raw, ref = stack('coupon_features_raw'), stack('ref_coupons')
    aligned = np.empty_like(raw, dtype=float)
    for channel in np.unique(lab):
        take = lab == channel
        center = np.nanmedian(np.nanmean(ref[take], axis=0))
        scale = np.nanmedian(np.nanstd(ref[take], axis=0, ddof=0))
        aligned[take] = (raw[take] - center) / scale

    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = []
    for channel, group in cells:
        values = aligned[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), counts,
                               out=np.full(raw.shape[1], np.nan), where=counts > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(means, axis=0))
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, aligned, stack('strain_0')


class _Model:
    def __init__(self, dataset, past, hold, aligned):
        self.dt = np.asarray(dataset['past_dt'])
        self.t = np.asarray(dataset['eval_times'])
        self.age = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.past, self.hold = past, hold
        damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)**2 @ self.dt
        active = (damage > 0) & (dataset['shear_banding_index'] <= 0.50)
        self.act = np.where(active, np.tanh(damage), 0.0)
        self.flow = np.broadcast_to(np.r_[np.log1p(self.t), 1.0], (len(hold), len(self.t) + 1))
        self.drift = self.flow[:, :, None] * np.column_stack((np.ones(len(hold)), aligned))[:, None, :]

    def design(self, logtau, indices=slice(None)):
        tau = np.exp(logtau)
        kernel = np.exp(-self.age[:, None] / tau) * -np.expm1(-self.dt[:, None] / tau)
        z = self.past[indices] @ kernel
        hold, act = self.hold[indices], self.act[indices]
        response = np.vstack((-np.expm1(-self.t[:, None] / tau), 1.0 / tau))
        memory = (hold[:, None] - z)[:, None, :] * response
        crystal = (act * hold)[:, None] * response[:, 0]
        arrest = (act * z[:, 0])[:, None] * self.flow[indices]
        return np.concatenate((memory, self.drift[indices], crystal[:, :, None], arrest[:, :, None]), axis=2)


def _fit(model, y, indices):
    y = y[indices]
    shape = y.shape
    target = y.ravel()
    ncols = 9
    if not np.any(model.act[indices]):
        ncols = 7

    def project(logtau, weight):
        design = model.design(logtau, indices)[:, :, :ncols].reshape(-1, ncols)
        coeff = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)[0]
        return np.pad(coeff, (0, 9 - ncols)), (design @ coeff - target).reshape(shape)

    def optimize(start, weight):
        def residual(logtau):
            return project(logtau, weight)[1].ravel() * weight
        out = least_squares(residual, start, bounds=(np.log(0.02), np.log(100.0)),
                            diff_step=1e-4, max_nfev=70, ftol=1e-8)
        return out.x, *project(out.x, weight)

    starts = np.log([(0.25, 3), (0.7, 5), (1.5, 12), (3, 0.25), (5, 0.7), (12, 1.5)])
    fits = [optimize(start, np.ones(target.size)) for start in starts]
    initial = min(fits, key=lambda fit: np.sum(fit[2]**2))[2]
    sigma = [max(1e-8, 1.4826 * np.median(np.abs(r - np.median(r))))
             for r in (initial[:, :-1], initial[:, -1])]
    scale = np.r_[np.repeat(sigma[0], shape[1] - 1), sigma[1]]

    def objective(fit):
        return np.log1p((fit[2] / (2.385 * scale))**2).sum()

    candidates = []
    for inc in (True, False):
        branch = [fit for fit in fits if (fit[0][0] < fit[0][1]) == inc]
        if branch:
            fit = min(branch, key=objective)
            for _ in range(6):
                weight = (1 / (scale * np.sqrt(1 + (fit[2] / (2.385 * scale))**2))).ravel()
                fit = optimize(fit[0], weight)
            candidates.append(fit)
    logtau, coeff, _ = min(candidates, key=objective)
    return logtau, coeff


def solve_alpha_m97_c01_batch_04(dataset: dict) -> dict:
    past, hold, aligned, strain0 = _inputs(dataset)
    model = _Model(dataset, past, hold, aligned)
    train = dataset['train']
    n = len(train['lab_id'])
    y = np.column_stack((np.asarray(train['strain_curves_obs']) - strain0[:n, None],
                         train['instant_rate_obs']))
    # Fit the response law on measured covariates; imputation is for prediction.
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    fit_rows = np.flatnonzero(complete)
    all_rows = np.arange(n)
    design = model.design(np.log([0.7, 5.0]), fit_rows)
    ncols = 9 if np.any(model.act[:n]) else 7
    if not len(fit_rows) or np.linalg.matrix_rank(design[:, :, :ncols].reshape(-1, ncols)) < ncols:
        fit_rows = all_rows
    logtau, coeff = _fit(model, y, fit_rows)
    prediction = model.design(logtau, slice(n, None)) @ coeff
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': coeff[:7],
            'regime_mode_weights': coeff[7:], 'predicted_strain': prediction[:, :-1] + strain0[n:, None],
            'initial_rate_pred': prediction[:, -1]}
