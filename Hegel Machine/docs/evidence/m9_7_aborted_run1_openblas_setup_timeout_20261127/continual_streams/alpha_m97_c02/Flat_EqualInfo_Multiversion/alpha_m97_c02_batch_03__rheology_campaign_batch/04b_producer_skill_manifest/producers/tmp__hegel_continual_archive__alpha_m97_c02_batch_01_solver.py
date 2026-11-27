"""Estimate a calibrated two-branch rheology model for the plateau campaign."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _archived_calibration(channel):
    if not _ARCHIVE.is_file():
        return None
    with _ARCHIVE.open() as stream:
        history = json.load(stream)
    for table in reversed(history):
        take = np.asarray(table['lab_id']) == channel
        x = np.asarray(table['cmd_stress'])[take]
        if np.unique(x).size >= 2:
            return np.polyfit(x, np.asarray(table['true_stress'])[take], 1)
    return None


def _calibrate(table):
    lab, cmd, true = (np.asarray(table[key]) for key in ('lab_id', 'cmd_stress', 'true_stress'))
    result = np.empty((4, 2))
    for channel in range(4):
        take = (lab == channel) & np.isfinite(cmd) & np.isfinite(true)
        x, y = cmd[take], true[take]
        i, j = np.triu_indices(x.size, k=1)
        distinct = x[i] != x[j]
        i, j = i[distinct], j[distinct]
        spike = False
        if i.size:
            gain = np.median((y[i] - y[j]) / (x[i] - x[j]))
            offsets = y - gain*x
            clean = np.abs(offsets - np.median(offsets)) <= 3.0
            spike = not np.all(clean)
        else:
            clean = np.ones(x.size, dtype=bool)
        archive = _archived_calibration(channel) if x.size < 2 or spike else None
        if np.count_nonzero(clean) >= 4 and np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        elif archive is not None:
            result[channel] = archive
        elif np.unique(x[clean]).size >= 2:
            result[channel] = np.polyfit(x[clean], y[clean], 1)
        else:
            result[channel] = (1.0, np.median(y - x) if y.size else 0.0)
    return result


def _prepare(dataset):
    train, test = dataset['train'], dataset['test']
    keys = ('lab_id', 'stratum', 'coupon_features_raw', 'ref_coupons',
            'past_stress_cmd', 'stress_hold_cmd', 'strain_0')
    lab, stratum, x, ref, past, hold, strain0 = (
        np.concatenate((train[key], test[key])) for key in keys
    )
    x = x.astype(float, copy=False)
    for channel in np.unique(lab):
        mask = lab == channel
        panel = ref[mask]
        panel = panel[:, np.any(np.isfinite(panel), axis=0)]
        center = np.median(np.nanmean(panel, axis=0))
        scale = np.median(np.nanstd(panel, axis=0, ddof=0))
        x[mask] = (x[mask] - center) / scale

    # Each observed lab/stratum cell contributes one column mean, not one
    # vote per coupon; compute the means after panel alignment.
    cells = np.unique(np.column_stack((lab, stratum)), axis=0)
    means = np.full((len(cells), x.shape[1]), np.nan)
    for row, (channel, group) in enumerate(cells):
        values = x[(lab == channel) & (stratum == group)]
        counts = np.isfinite(values).sum(axis=0)
        np.divide(np.nansum(values, axis=0), counts, out=means[row], where=counts > 0)
    fill = np.nanmean(means, axis=0)
    x = np.where(np.isfinite(x), x, fill)
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    return (gain[:, None]*past + bias[:, None], gain*hold + bias,
            np.column_stack((np.ones(len(x)), x)), strain0)


def _activity(dataset, past, dt):
    if dataset['shear_banding_index'] > 0.50:
        return np.zeros(len(past))
    excess = np.maximum(0., np.abs(past) - dataset['crit_stress'])
    return np.tanh(excess**2 @ dt)


def _basis(log_taus, past, hold, features, dt, times, act, plateau):
    taus = np.exp(log_taus)
    age = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    a, q, r = age[:, None]/taus, dt[:, None]/taus, times[:, None]/taus
    ea, eq, er = np.exp(-a), np.exp(-q), np.exp(-r)
    memory = ea * -np.expm1(-q)
    z = past @ memory
    dz = past @ (a*memory - q*ea*eq)
    gap = hold[:, None] - z
    step = -np.expm1(-r)
    prony = gap[:, None, :]*step
    phi = np.broadcast_to(np.log1p(times) if plateau else times, (len(past), len(times)))
    drive = act * hold
    curve = np.concatenate((prony, phi[:, :, None]*features[:, None, :],
                            (drive[:, None]*step[:, 0])[..., None],
                            ((act*z[:, 0])[:, None]*phi)[..., None]), axis=2)
    rate = np.column_stack((gap/taus, features, drive/taus[0], act*z[:, 0]))
    dc = -dz[:, None, :]*step - gap[:, None, :]*r*er
    dr = (-dz - gap)/taus
    dc_extra = np.stack((-drive[:, None]*(r*er)[:, 0], (act*dz[:, 0])[:, None]*phi), axis=-1)
    dr_extra = np.column_stack((-drive/taus[0], act*dz[:, 0]))
    return curve, rate, np.concatenate((dc, dc_extra), axis=-1), np.column_stack((dr, dr_extra))


def _fit(past, hold, features, dt, times, act, curves, rates, plateau, scales=(0.025, 0.025)):
    target = np.r_[curves.ravel()/scales[0], rates/scales[1]]

    def design(log_taus):
        c, r, dc, dr = _basis(log_taus, past, hold, features, dt, times, act, plateau)
        return (np.vstack((c.reshape(-1, 9)/scales[0], r/scales[1])),
                np.vstack((dc.reshape(-1, 4)/scales[0], dr/scales[1])))

    def residual(params):
        matrix, _ = design(params[:2])
        return np.einsum('ij,j->i', matrix, params[2:]) - target

    def jacobian(params):
        matrix, d = design(params[:2])
        fast = d[:, 0]*params[2] + d[:, 2]*params[9] + d[:, 3]*params[10]
        return np.column_stack((fast, d[:, 1]*params[3], matrix))

    lo = np.r_[np.log([0.01, 0.01]), np.full(9, -np.inf)]
    hi = np.r_[np.log([100., 100.]), np.full(9, np.inf)]
    best = None
    for pair in ((0.45, 2.4), (0.20, 1.0), (0.20, 6.0), (1.0, 6.0), (0.09, 2.4), (0.45, 15.0)):
        log_taus = np.log(pair)
        matrix, _ = design(log_taus)
        weights = np.linalg.lstsq(matrix, target, rcond=None)[0]
        fit = least_squares(residual, np.r_[log_taus, weights], jac=jacobian,
                            bounds=(lo, hi), loss='cauchy', x_scale='jac', max_nfev=130)
        if best is None or fit.cost < best.cost:
            best = fit
    params = best.x.copy()
    if not np.any(act) and params[0] > params[1]:
        params[[0, 1, 2, 3]] = params[[1, 0, 3, 2]]
    return params[:2], params[2:]


def solve_alpha_m97_c02_batch_01(dataset: dict) -> dict:
    past, hold, features, strain0 = _prepare(dataset)
    dt, times = (np.asarray(dataset[k], dtype=float) for k in ('past_dt', 'eval_times'))
    # The campaign selects the terminal basis; work damage gates only the two extra terms.
    plateau = dataset['shear_banding_index'] <= 0.50
    act = _activity(dataset, past, dt)
    train = dataset['train']
    n = len(train['lab_id'])
    curves = np.asarray(train['strain_curves_obs']) - strain0[:n, None]
    taus, weights = _fit(past[:n], hold[:n], features[:n], dt, times, act[:n],
                         curves, np.asarray(train['instant_rate_obs']), plateau)
    curve, rate, _, _ = _basis(taus, past[n:], hold[n:], features[n:], dt, times, act[n:], plateau)
    return {'estimated_taus': np.exp(taus), 'hereditary_weights': weights[:7],
            'regime_mode_weights': weights[7:], 'predicted_strain': strain0[n:, None] + curve @ weights,
            'initial_rate_pred': rate @ weights}
