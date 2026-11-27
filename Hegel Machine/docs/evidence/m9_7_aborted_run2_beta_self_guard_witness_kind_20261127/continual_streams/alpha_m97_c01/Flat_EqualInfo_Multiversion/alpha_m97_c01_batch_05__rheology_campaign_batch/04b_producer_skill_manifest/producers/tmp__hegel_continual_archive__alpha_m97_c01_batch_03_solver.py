"""Variable-projection solver for calibrated two-branch jammed-slip creep."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_HISTORY = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]


def _history_affine(history, channel):
    for sweep in reversed(history):
        use = np.asarray(sweep['lab_id']) == channel
        x, y = (np.asarray(sweep[k])[use] for k in ('cmd_stress', 'true_stress'))
        if len(x) >= 2 and np.ptp(x) > 0:
            return _affine(x, y)
    return None


def _calibrate(dataset):
    rows = dataset['calibration_load_cells']
    lab, cmd, true = (np.asarray(rows[k]) for k in ('lab_id', 'cmd_stress', 'true_stress'))
    coefficients = np.empty((4, 2))
    history = None
    for channel in range(4):
        selected = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[selected], true[selected]
        if len(x) == 3:
            # With three rows, a line through a spike can look as plausible as
            # the clean pair. Screen all omissions; use history only for an
            # apparent EMI event, allowing 0.01 for amplified readout noise.
            loo = []
            for i in range(3):
                keep = np.arange(3) != i
                if np.ptp(x[keep]) > 0:
                    gain, bias = _affine(x[keep], y[keep])
                    loo.append(abs(gain * x[i] + bias - y[i]))
            if max(loo, default=0.0) > 2.99 and _HISTORY.is_file():
                if history is None:
                    history = json.loads(_HISTORY.read_text())
                prior = _history_affine(history, channel)
                if prior is not None:
                    gain = prior[0]
                    intercepts = y - gain * x
                    bias = np.median(intercepts)
                    if np.max(np.abs(intercepts - bias)) > 3.0:
                        coefficients[channel] = gain, bias
                        continue
        emi = False
        if len(x) >= 3:
            i, j = np.triu_indices(len(x), 1)
            distinct = x[i] != x[j]
            i, j = i[distinct], j[distinct]
            if i.size:
                slopes = (y[j] - y[i]) / (x[j] - x[i])
                errors = np.abs(y[i, None] + slopes[:, None] * (x - x[i, None]) - y)
                best = np.lexsort((np.sum(np.minimum(errors, 3.0), axis=1), np.median(errors, axis=1)))[0]
                core = errors[best] <= max(8 * np.median(errors[best]), 1e-10)
                gain, bias = _affine(x[core], y[core])
                residual = np.abs(gain * x + bias - y)
                emi = bool(np.any(residual > 3.0))
                x, y = x[residual <= 3.0], y[residual <= 3.0]
        if len(x) >= 2 and np.ptp(x) > 0:
            coefficients[channel] = _affine(x, y)
        elif (selected.sum() < 2 or emi) and _HISTORY.is_file():
            if history is None:
                history = json.loads(_HISTORY.read_text())
            prior = _history_affine(history, channel)
            if prior is None:
                coefficients[channel] = 1.0, np.median(y - x) if x.size else 0.0
            else:
                gain, bias = prior
                coefficients[channel] = gain, np.median(y - gain * x) if x.size else bias
        else:
            coefficients[channel] = 1.0, np.median(y - x) if x.size else 0.0
    return coefficients


def _inputs(dataset):
    splits = (dataset['train'], dataset['test'])

    def stack(key):
        return np.concatenate([s[key] for s in splits], axis=0)

    lab, strata = stack('lab_id'), stack('stratum')
    raw, ref = stack('coupon_features_raw'), stack('ref_coupons')
    aligned = np.empty_like(raw, dtype=float)
    for channel in np.unique(lab):
        m = lab == channel
        center = np.nanmedian(np.nanmean(ref[m], axis=0))
        scale = np.nanmedian(np.nanstd(ref[m], axis=0, ddof=0))
        aligned[m] = (raw[m] - center) / scale
    cells = np.unique(np.column_stack((lab, strata)), axis=0)
    means = []
    for c, s in cells:
        v = aligned[(lab == c) & (strata == s)]
        count = np.isfinite(v).sum(axis=0)
        means.append(np.divide(np.nansum(v, axis=0), count,
                               out=np.full(v.shape[1], np.nan), where=count > 0))
    fill = np.nanmean(means, axis=0)
    aligned = np.where(np.isfinite(aligned), aligned, fill)
    gain, bias = _calibrate(dataset)[lab].T
    past = gain[:, None] * stack('past_stress_cmd') + bias[:, None]
    hold = gain * stack('stress_hold_cmd') + bias
    return past, hold, aligned, stack('strain_0')


class _Model:
    def __init__(self, dt, t, past, hold, x):
        self.dt, self.t = np.asarray(dt), np.asarray(t)
        self.age = np.r_[np.cumsum(self.dt[:0:-1])[::-1], 0.0]
        self.past, self.hold = past, hold
        terminal = np.r_[self.t / (1.0 + 0.85 * self.t), 1.0]
        slip = np.sign(hold) * np.sqrt(np.abs(hold))
        features = np.column_stack((np.ones(len(x)), x, slip))
        self.flow = features[:, None, :] * terminal[None, :, None]

    def design(self, logtau):
        tau = np.exp(logtau)
        kernel = np.exp(-self.age[:, None] / tau) * -np.expm1(-self.dt[:, None] / tau)
        gap = self.hold[:, None] - self.past @ kernel
        response = np.vstack((-np.expm1(-self.t[:, None] / tau), 1.0 / tau))
        memory = gap[:, None, :] * response
        interaction = (gap[:, 0] - gap[:, 1])[:, None] * response[:, 0]
        return np.concatenate((memory, self.flow, interaction[:, :, None]), axis=2)


def _scales(residual):
    groups = residual[:, :-1], residual[:, -1:]
    return np.array([max(1e-8, 1.4826 * np.median(np.abs(g - np.median(g)))) for g in groups])


def _fit(model, y, indices=None):
    if indices is None:
        indices = np.arange(len(y))
    y = y[indices]
    shape = y.shape
    target = y.ravel()
    bounds = np.log([0.025, 100.0])

    def fit_linear(logtau, weight):
        design = model.design(logtau)[indices].reshape(-1, 9)
        scaled = design * weight[:, None]
        # Normalize columns before profiling: near-identical relaxation times
        # can produce tiny, but physically meaningful, interaction columns.
        norm = np.maximum(np.sqrt(np.sum(scaled * scaled, axis=0)), 1e-15)
        w = np.linalg.lstsq(scaled / norm, target * weight, rcond=1e-11)[0] / norm
        return w, np.einsum('ij,j->i', design, w) - target

    def optimize(start, weight):
        result = least_squares(lambda p: fit_linear(p, weight)[1] * weight,
                               start, bounds=(bounds[0], bounds[1]),
                               diff_step=1e-4, max_nfev=55, ftol=2e-8)
        w, residual = fit_linear(result.x, weight)
        return result.x, w, residual.reshape(shape)

    weight = np.ones(target.size)
    starts = np.log([(0.25, 1.4), (0.55, 3.5), (1.2, 6.5), (0.4, 12.),
                     (1.4, 0.25), (3.5, 0.55), (6.5, 1.2), (12., 0.4)])
    ordinary = [optimize(start, weight) for start in starts]
    _, _, initial = min(ordinary, key=lambda fit: np.sum(fit[2] ** 2))
    sigma = _scales(initial)
    scale = np.broadcast_to(np.r_[np.repeat(sigma[0], shape[1] - 1), sigma[1]], shape)

    def irls_weights(residual):
        u = residual / (2.385 * scale)
        return (1.0 / (scale * np.sqrt(1.0 + u * u))).ravel()

    def objective(fit):
        return np.sum(np.log1p((fit[2] / (2.385 * scale)) ** 2))

    weight = irls_weights(initial)
    candidates = [optimize(start, weight) for start in starts]
    # Keep the best basin from each branch order through robust refinement.
    refined = []
    for increasing in (True, False):
        branch = [fit for fit in candidates if (fit[0][0] < fit[0][1]) == increasing]
        if branch:
            fit = min(branch, key=objective)
            for _ in range(5):
                fit = optimize(fit[0], irls_weights(fit[2]))
            refined.append(fit)
    tau, weights, _ = min(refined, key=objective)
    return tau, weights


def solve_alpha_m97_c01_batch_03(dataset: dict) -> dict:
    past, hold, x, strain0 = _inputs(dataset)
    model = _Model(dataset['past_dt'], dataset['eval_times'], past, hold, x)
    train = dataset['train']
    n = len(train['lab_id'])
    y = np.column_stack((np.asarray(train['strain_curves_obs']) - strain0[:n, None],
                         train['instant_rate_obs']))
    logtau, weights = _fit(model, y)
    prediction = model.design(logtau)[n:] @ weights
    return {'estimated_taus': np.exp(logtau), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': prediction[:, :-1] + strain0[n:, None],
            'initial_rate_pred': prediction[:, -1]}
