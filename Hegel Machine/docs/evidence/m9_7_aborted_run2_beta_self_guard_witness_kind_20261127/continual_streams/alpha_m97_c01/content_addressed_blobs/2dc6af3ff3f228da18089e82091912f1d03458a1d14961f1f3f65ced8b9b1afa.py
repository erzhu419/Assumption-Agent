"""Two-branch hereditary creep in the jammed-slip regime."""
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
        if clean.sum() >= 4 and np.ptp(x[clean]) > 0:
            coefficients[channel] = _affine(x[clean], y[clean])
            continue
        if len(x) < 2 or emi:
            gain, bias = prior(channel)
            # The archived sweep supplies an absolute reference: an EMI burst
            # can move most of a sparse sweep in the same direction.
            clean = np.abs(y - (gain * x + bias)) <= 3.0
        if clean.sum() >= 2 and np.ptp(x[clean]) > 0:
            coefficients[channel] = _affine(x[clean], y[clean])
        elif len(x) < 2 or emi:
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
        self.flow = np.r_[self.t / (1.0 + 0.85 * self.t), 1.0]
        slip = np.sign(hold) * np.sqrt(np.abs(hold))
        self.drift = self.flow[None, :, None] * np.column_stack(
            (np.ones(len(hold)), aligned, slip))[:, None, :]

    def design(self, logtau, indices=slice(None)):
        tau = np.exp(logtau)
        kernel = np.exp(-self.age[:, None] / tau) * -np.expm1(-self.dt[:, None] / tau)
        gap = self.hold[indices, None] - self.past[indices] @ kernel
        response = np.vstack((-np.expm1(-self.t[:, None] / tau), 1.0 / tau))
        memory = gap[:, None, :] * response
        exchange = (gap[:, :1] - gap[:, 1:]) * response[:, 0]
        return np.concatenate((memory, self.drift[indices], exchange[:, :, None]), axis=2)


def _fit(model, y, rows):
    y = y[rows]
    ncols = 9
    lower, upper = np.log([0.025, 100.0])

    def optimize(logtau, weights):
        root = np.sqrt(weights.ravel())
        target = y.ravel() * root

        def project(t):
            design = model.design(t, rows).reshape(-1, ncols)
            a = design * root[:, None]
            coeff = np.linalg.lstsq(a, target, rcond=1e-10)[0]
            return coeff, design @ coeff - y.ravel()

        result = least_squares(lambda t: project(t)[1] * root, logtau,
                               bounds=(lower, upper), diff_step=1e-4,
                               max_nfev=65, ftol=1e-9)
        coeff, residual = project(result.x)
        return result.x, coeff, residual.reshape(y.shape)

    # Start with a broad clipping scale, then estimate endpoint noise from the
    # central residuals as isolated measurement spikes are downweighted.
    scale0 = np.maximum(np.median(np.abs(y - np.median(y, axis=0)), axis=0), 1e-6)
    weights0 = 1 / (1 + ((y - np.median(y, axis=0)) / (8 * scale0)) ** 2)
    candidates = []
    for seed in ([0.35, 3.0], [0.85, 6.0], [2.0, 0.45], [6.0, 0.85]):
        logtau, coeff, err = optimize(np.log(seed), weights0)
        for _ in range(7):
            scale = np.maximum(1.4826 * np.median(np.abs(err - np.median(err, axis=0)), axis=0), 1e-7)
            weights = 1 / (1 + (err / (3 * scale)) ** 2)
            logtau, coeff, err = optimize(logtau, weights)
        score = np.mean(np.sort(np.abs(err), axis=0)[:int(0.85 * len(y))] ** 2)
        candidates.append((score, logtau[0] >= logtau[1], logtau, coeff))
    # Branch zero is the fast relaxation mode.
    ordered = [entry for entry in candidates if not entry[1]]
    best = min(ordered or candidates, key=lambda entry: entry[0])
    return best[2], best[3]


def solve_alpha_m97_c01_batch_06(dataset: dict) -> dict:
    past, hold, aligned, strain0 = _inputs(dataset)
    model = _Model(dataset, past, hold, aligned)
    train = dataset['train']
    n = len(train['lab_id'])
    y = np.column_stack((np.asarray(train['strain_curves_obs']) - strain0[:n, None],
                         train['instant_rate_obs']))
    rows = np.flatnonzero(np.isfinite(train['coupon_features_raw']).all(axis=1))
    if len(rows) < 9 or np.linalg.matrix_rank(model.design(np.log([0.7, 5.0]), rows).reshape(-1, 9)) < 9:
        rows = np.arange(n)
    logtau, coeff = _fit(model, y, rows)
    prediction = model.design(logtau, slice(n, None)) @ coeff
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': coeff[:7],
            'regime_mode_weights': coeff[7:], 'predicted_strain': prediction[:, :-1] + strain0[n:, None],
            'initial_rate_pred': prediction[:, -1]}
