"""Calibrated two-branch Prony rheology with a damage-gated logarithmic plateau."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _channel(sweep, lab):
    x = np.asarray(sweep['cmd_stress'], dtype=float)
    y = np.asarray(sweep['true_stress'], dtype=float)
    use = (np.asarray(sweep['lab_id']) == lab) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _affine(x, y):
    return np.linalg.lstsq(np.column_stack((x, np.ones_like(x))), y, rcond=None)[0]


def _clean(x, y):
    good = np.ones(len(x), dtype=bool)
    if len(x) < 3:
        return good
    # Select the most coherent majority before measuring EMI residuals.
    best = (np.inf, np.inf)
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            if x[j] == x[i]:
                continue
            g = (y[j] - y[i]) / (x[j] - x[i])
            r = np.abs((y - y[i]) - g*(x - x[i]))
            score = (np.median(r), np.mean(np.sort(r)[:len(x)-1]))
            if score < best:
                best, good = score, r <= 3.0
    return good


def _calibrate(sweep):
    result = np.empty((4, 2))
    history = None
    for lab in range(4):
        x, y = _channel(sweep, lab)
        good = _clean(x, y)
        if good.sum() >= 4 or (good.all() and len(x) >= 2):
            if np.ptp(x[good]) == 0:
                raise ValueError(f'No identifiable load-cell calibration for lab {lab}')
            result[lab] = _affine(x[good], y[good])
            continue
        if history is None:
            history = json.loads(_ARCHIVE.read_text())
        for archived_sweep in reversed(history):
            ax, ay = _channel(archived_sweep, lab)
            clean = _clean(ax, ay)
            if clean.sum() >= 2 and np.ptp(ax[clean]) > 0:
                result[lab] = _affine(ax[clean], ay[clean])
                break
        else:
            raise ValueError(f'No usable calibration for lab {lab}')
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']

    def join(key):
        return np.concatenate((train[key], test[key]), axis=0)

    lab, stratum = join('lab_id'), join('stratum')
    refs, raw = join('ref_coupons'), join('coupon_features_raw')
    aligned = np.empty_like(raw, dtype=float)
    for c in range(4):
        use = lab == c
        ref = refs[use]
        center = np.nanmedian(np.nanmean(ref, axis=0))
        scale = np.nanmedian(np.nanstd(ref, axis=0, ddof=0))
        aligned[use] = (raw[use] - center) / scale
    # Give each (lab, stratum) cell equal weight, regardless of its size.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = []
    for c, stratum_id in cells:
        values = aligned[(lab == c) & (stratum == stratum_id)]
        count = np.isfinite(values).sum(axis=0)
        means.append(np.divide(np.nansum(values, axis=0), count,
                               out=np.full(raw.shape[1], np.nan), where=count > 0))
    aligned = np.where(np.isfinite(aligned), aligned, np.nanmean(means, axis=0))
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    dt = np.asarray(dataset['past_dt'])
    damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.)**2 @ dt
    act = np.tanh(damage) if dataset['shear_banding_index'] <= 0.50 else np.zeros_like(damage)
    return aligned, past, hold, act


def _design(aligned, past, hold, act, dt, times, taus, low_shear):
    # Unroll the exact recurrence: each step's input is attenuated by the
    # remaining history duration, not by the step's own duration twice.
    remaining = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    memory = past @ (-np.expm1(-dt[:, None]/taus) * np.exp(-remaining[:, None]/taus))
    gap = hold[:, None] - memory
    relax = -np.expm1(-times[:, None]/taus)
    drift = np.column_stack((np.ones(len(hold)), aligned))
    terminal = np.log1p(times) if low_shear else times
    curve = np.empty((len(hold), len(times), 9))
    curve[:, :, :2] = gap[:, None, :]*relax
    curve[:, :, 2:7] = drift[:, None, :]*terminal[None, :, None]
    curve[:, :, 7] = (act*hold)[:, None]*relax[:, 0]
    curve[:, :, 8] = (act*memory[:, 0])[:, None]*terminal
    rate = np.column_stack((gap/taus, drift, act*hold/taus[0], act*memory[:, 0]))
    return curve, rate


def _fit(aligned, past, hold, act, dt, times, y, r, low_shear):
    # Profile the nine linear weights at each pair of relaxation times.
    # Cauchy weights suppress individual spikes, not the rest of their curve.
    response = np.r_[y.ravel(), r]
    scales = np.r_[np.full(y.size, .055), np.full(len(r), .13)]
    logtau = np.log([.6, 3.5])
    sw = np.ones_like(response)

    def matrix(p):
        curve, rate = _design(aligned, past, hold, act, dt, times, np.exp(p), low_shear)
        return np.vstack((curve.reshape(-1, 9), rate))

    for step in range(25):
        target = sw*response

        def profile(p, coefficients=False):
            design = sw[:, None]*matrix(p)
            weights = np.linalg.lstsq(design, target, rcond=None)[0]
            if coefficients:
                return weights
            return np.einsum('ij,j->i', design, weights) - target

        logtau = least_squares(profile, logtau, bounds=(np.log(.035), np.log(60.)),
                               max_nfev=45, ftol=1e-8, xtol=1e-8).x
        weights = profile(logtau, coefficients=True)
        residual = np.einsum('ij,j->i', matrix(logtau), weights) - response
        updated = 1/np.hypot(scales*(3 if step < 3 else 1), residual)
        if step > 4 and np.max(np.abs(updated/sw - 1)) < 1e-4:
            break
        sw = updated
    return np.exp(logtau), weights


def solve_alpha_m97_c02_batch_04(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    n = len(train['lab_id'])
    data = _prepare(dataset)
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    low_shear = dataset['shear_banding_index'] <= .50
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(*(a[:n][complete] for a in data), dt, times,
                         y[complete], train['instant_rate_obs'][complete], low_shear)
    curve, rate = _design(*(a[n:] for a in data), dt, times, taus, low_shear)
    return {'estimated_taus': taus, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
