import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _line(x, y):
    xc = x - x.mean()
    gain = xc @ (y - y.mean()) / (xc @ xc)
    return gain, y.mean() - gain * x.mean()


def _spikes(x, y):
    if len(x) < 3:
        return np.zeros(len(x), dtype=bool)
    i, j = np.triu_indices(len(x), 1)
    distinct = x[i] != x[j]
    i, j = i[distinct], j[distinct]
    if not len(i):
        return np.zeros(len(x), dtype=bool)
    gains = (y[j] - y[i]) / (x[j] - x[i])
    offsets = y[:, None] - x[:, None] * gains
    errors = np.abs(offsets - np.median(offsets, axis=0))
    best = np.argmin(np.median(errors, axis=0))
    return errors[:, best] > 3.0


def _prior_line(channel):
    for sweep in reversed(json.loads(_ARCHIVE.read_text())):
        row = np.asarray(sweep['lab_id']) == channel
        x = np.asarray(sweep['cmd_stress'])[row]
        y = np.asarray(sweep['true_stress'])[row]
        keep = ~_spikes(x, y)
        if np.unique(x[keep]).size >= 2:
            return _line(x[keep], y[keep])
    raise ValueError(f'No usable calibration sweep for channel {channel}')


def _calibrate(x, y, channel):
    if len(x) >= 4:
        keep = ~_spikes(x, y)
        if keep.sum() >= 4 or keep.all():
            return _line(x[keep], y[keep])
    else:
        keep = np.ones(len(x), dtype=bool)
        if len(x) == 3:
            # Two surviving points cannot identify which of three rows spiked.
            # Look for a >3 discrepancy before consulting the prior sweep.
            errors = [abs(y[k] - np.polyval(_line(np.delete(x, k), np.delete(y, k)), x[k]))
                      for k in range(3) if np.unique(np.delete(x, k)).size == 2]
            slope, bias = _line(x, y)
            if not errors or max(errors) <= 3 or np.max(np.abs(y - slope*x - bias)) < .01:
                return slope, bias
        elif len(x) == 2:
            return _line(x, y)
    gain, bias = _prior_line(channel)
    if len(x) >= 3 and keep.all():
        keep = np.abs(y - (gain*x + bias)) <= 3.0
    if np.unique(x[keep]).size >= 2:
        clean_x, clean_y = x[keep], y[keep]
        fitted_gain, fitted_bias = _line(clean_x, clean_y)
        # With fewer than four clean readings after EMI rejection, borrow
        # the archived slope only if supported by the in-batch contrasts.
        # Keep the stress level anchored to the current measurements.
        x_centered = clean_x - clean_x.mean()
        if keep.sum() < 4:
            contrasts = clean_y - clean_y.mean() - gain * x_centered
            if np.ptp(contrasts) <= 0.001:
                return gain, clean_y.mean() - gain * clean_x.mean()
        return fitted_gain, fitted_bias
    return gain, bias


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    cal = dataset['calibration_load_cells']
    lab = np.r_[train['lab_id'], test['lab_id']]
    stratum = np.r_[train['stratum'], test['stratum']]
    ref = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    past = np.vstack((train['past_stress_cmd'], test['past_stress_cmd'])).astype(float)
    hold = np.r_[train['stress_hold_cmd'], test['stress_hold_cmd']].astype(float)
    for c in np.unique(lab):
        rows = np.asarray(cal['lab_id']) == c
        gain, bias = _calibrate(np.asarray(cal['cmd_stress'])[rows], np.asarray(cal['true_stress'])[rows], c)
        take = lab == c
        past[take] = gain * past[take] + bias
        hold[take] = gain * hold[take] + bias
        panel = ref[take]
        x[take] = (x[take] - np.nanmedian(np.nanmean(panel, axis=0))) / np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
    cells = []
    for c in np.unique(lab):
        for stratum_id in np.unique(stratum[lab == c]):
            take = (lab == c) & (stratum == stratum_id)
            valid = np.isfinite(x[take]).sum(axis=0)
            mean = np.divide(np.nansum(x[take], axis=0), valid, out=np.full(x.shape[1], np.nan), where=valid != 0)
            cells.append((c, take, mean))
    for c, take, mean in cells:
        if np.isnan(mean).any():
            with np.errstate(invalid='ignore', divide='ignore'):
                local = np.array([v for lab_id, _, v in cells if lab_id == c])
                count = np.isfinite(local).sum(axis=0)
                local = np.divide(np.nansum(local, axis=0), count, out=np.zeros(x.shape[1]), where=count != 0)
            mean = np.where(np.isnan(mean), local, mean)
        x[take] = np.where(np.isnan(x[take]), mean, x[take])
    return np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times']), past, hold, x



def _design(tau, dt, t, past, hold, x, active, low_shear):
    decay = np.exp(-dt[None, :] / tau[:, None])
    step = -np.expm1(-dt[None, :] / tau[:, None])
    memory = np.zeros((len(x), 2))
    for s in range(len(dt)):
        memory = decay[:, s] * memory + step[:, s] * past[:, s, None]
    gaps = hold[:, None] - memory
    transient = -np.expm1(-t[:, None] / tau)
    terminal = np.where(low_shear[:, None], np.log1p(t), t)
    drift = np.column_stack((np.ones(len(x)), x))
    curves = np.concatenate((
        gaps[:, None, :] * transient[None, :, :],
        terminal[:, :, None] * drift[:, None, :],
        (active * hold)[:, None, None] * transient[None, :, :1],
        (active * memory[:, 0])[:, None, None] * terminal[:, :, None],
    ), axis=2)
    rate = np.column_stack((gaps / tau, drift, active * hold / tau[0], active * memory[:, 0]))
    return curves, rate


def _activation(dataset, past, dt):
    damage = np.maximum(np.abs(past) - dataset['crit_stress'], 0.0)**2 @ dt
    # Low shear sets the terminal time law; damage activates the two mode terms.
    low_shear = np.broadcast_to(dataset['shear_banding_index'] <= 0.50, damage.shape)
    return np.where(low_shear, np.tanh(damage), 0.0), low_shear


def _fit(dt, t, past, hold, x, active, low_shear, yc, yr, complete):
    # Missing predictors induce a correlated drift error across all times of a
    # coupon. Estimate the shared coefficients from observed predictors.
    good_c = np.isfinite(yc) & complete[:, None]
    good_r = np.isfinite(yr) & complete
    y = np.r_[yc[good_c], yr[good_r]]

    def profile(log_tau, cutoff=None):
        curves, rate = _design(np.exp(log_tau), dt, t, past, hold, x, active, low_shear)
        a = np.vstack((curves[good_c], rate[good_r]))
        w = np.linalg.lstsq(a, y, rcond=None)[0]
        e = np.einsum('ij,j->i', a, w) - y
        for _ in range(16):
            q = (1.0 + (e / 0.02)**2)**(-0.25)
            w = np.linalg.lstsq(a*q[:, None], y*q, rcond=None)[0]
            e = np.einsum('ij,j->i', a, w) - y
        if cutoff is None:
            return e * np.sqrt(2.0 / (1.0 + np.hypot(1.0, e / 0.02))), w
        # A bounded loss rejects isolated acquisition spikes after the first fit.
        for _ in range(14):
            q = np.maximum(1.0 - (e/cutoff)**2, 0.0)
            w = np.linalg.lstsq(a*q[:, None], y*q, rcond=None)[0]
            e = np.einsum('ij,j->i', a, w) - y
        u2 = np.minimum((e/cutoff)**2, 1.0)
        objective = np.clip(e, -cutoff, cutoff) * np.sqrt(1.0 - u2 + u2*u2/3.0)
        return objective, w

    bounds = (np.log(0.025), np.log(100.0))
    pilot = least_squares(lambda u: profile(u)[0], np.log([0.55, 3.0]), bounds=bounds)
    final = least_squares(lambda u: profile(u, .075)[0], pilot.x, bounds=bounds)
    return np.exp(final.x), profile(final.x, .075)[1]


def solve_alpha_m97_c01_batch_04(dataset: dict) -> dict:
    train = dataset['train']
    dt, t, past, hold, x = _prepare(dataset)
    active, plateau = _activation(dataset, past, dt)
    n = len(train['lab_id'])
    strain0 = np.asarray(train['strain_0'])
    yc = np.asarray(train['strain_curves_obs']) - strain0[:, None]
    yr = np.asarray(train['instant_rate_obs'])
    complete = np.isfinite(train['coupon_features_raw']).all(axis=1)
    tau, weights = _fit(dt, t, past[:n], hold[:n], x[:n], active[:n], plateau[:n], yc, yr, complete)
    curves, rates = _design(tau, dt, t, past[n:], hold[n:], x[n:], active[n:], plateau[n:])
    return {'estimated_taus': tau, 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:],
            'predicted_strain': np.asarray(dataset['test']['strain_0'])[:, None] + curves @ weights,
            'initial_rate_pred': rates @ weights}
