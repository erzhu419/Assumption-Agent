"""Calibrated two-branch creep and jammed-slip inference."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

_ARCHIVE = Path('/tmp/hegel_continual_archive/archive_calibration_history.json')


def _points(rows, channel):
    x, y = (np.asarray(rows[k], float) for k in ('cmd_stress', 'true_stress'))
    use = (np.asarray(rows['lab_id']) == channel) & np.isfinite(x) & np.isfinite(y)
    return x[use], y[use]


def _screen(x, y):
    if len(x) < 3:
        return np.ones(len(x), dtype=bool)
    dx = x[:, None] - x
    dy = y[:, None] - y
    slopes = [np.median(b[a != 0] / a[a != 0])
              for a, b in zip(dx, dy) if np.any(a != 0)]
    slope = np.median(slopes) if slopes else 0.
    offset = y - slope*x
    return np.abs(offset - np.median(offset)) <= 3.


def _calibrate(rows):
    result = np.empty((4, 2))
    sweeps = None
    sparse = []
    max_gain = 1.5
    for c in range(4):
        x, y = _points(rows, c)
        clean = _screen(x, y)
        if clean.sum() >= 4 and np.ptp(x[clean]):
            result[c] = np.polyfit(x[clean], y[clean], 1)
            max_gain = max(max_gain, result[c, 0])
        else:
            sparse.append((c, x, y, clean))
    for c, x, y, clean in sparse:
        emi = not clean.all()
        if len(x) == 2:
            # A two-point fit has zero residual. Check the response contrast
            # against the positive load-cell gain range seen in this campaign.
            dx, dy = np.diff(x)[0], np.diff(y)[0]
            lo, hi = sorted((0., max_gain*dx))
            emi = dy < lo - 3. or dy > hi + 3.
        if len(x) >= 2 and not emi and np.ptp(x):
            result[c] = np.polyfit(x, y, 1)
            continue
        if len(x) >= 2 and not emi:
            raise ValueError(f'Calibration commands lack variation for channel {c}')
        if sweeps is None:
            sweeps = json.loads(_ARCHIVE.read_text())
        for sweep in reversed(sweeps):
            a, b = _points(sweep, c)
            good = _screen(a, b)
            if good.sum() >= 2 and np.ptp(a[good]):
                gain, bias = np.polyfit(a[good], b[good], 1)
                break
        else:
            raise ValueError(f'No usable prefix calibration for channel {c}')
        keep = np.abs(y - (gain*x + bias)) <= 3.
        if keep.sum() >= 2 and np.ptp(x[keep]):
            gain, bias = np.polyfit(x[keep], y[keep], 1)
        elif keep.any():
            bias = np.mean(y[keep] - gain*x[keep])
        result[c] = gain, bias
    return result


def _prepare(dataset):
    def join(key):
        return np.concatenate((dataset['train'][key], dataset['test'][key]))

    lab, stratum = join('lab_id'), join('stratum')
    gain, bias = _calibrate(dataset['calibration_load_cells'])[lab].T
    past = gain[:, None]*join('past_stress_cmd') + bias[:, None]
    hold = gain*join('stress_hold_cmd') + bias
    features, ref = join('coupon_features_raw').astype(float), join('ref_coupons')
    for channel in np.unique(lab):
        sel = lab == channel
        panel = ref[sel]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        center = np.median(np.nanmean(panel, axis=0))
        spread = np.median(np.nanstd(panel, axis=0, ddof=0))
        features[sel] = (features[sel] - center)/spread
    cells = [(lab == c) & (stratum == s)
             for c, s in np.unique(np.column_stack((lab, stratum)), axis=0)]
    means = np.array([np.nanmean(features[cell], axis=0) for cell in cells])
    balanced = np.nanmean(means, axis=0)
    for cell, mean in zip(cells, means):
        features[cell] = np.where(np.isnan(features[cell]),
                                  np.where(np.isnan(mean), balanced, mean), features[cell])
    return past, hold, features


def _design(tau, state, dt, times):
    past, hold, features = state
    decay = np.exp(-dt[:, None]/tau)
    transfer = -np.expm1(-dt[:, None]/tau)
    memory = np.zeros((len(hold), 2))
    for s in range(len(dt)):
        memory = decay[s]*memory + past[:, s, None]*transfer[s]
    gap = hold[:, None] - memory
    slip = np.sign(hold)*np.sqrt(np.abs(hold))
    drift = np.column_stack((np.ones(len(hold)), features, slip))
    mismatch = gap[:, 0] - gap[:, 1]
    creep = -np.expm1(-times[:, None]/tau)
    flow = times/(1. + .85*times)
    curve = np.concatenate((gap[:, None, :]*creep,
                            drift[:, None, :]*flow[None, :, None],
                            (mismatch[:, None]*creep[:, 0])[:, :, None]), axis=2)
    rate = np.column_stack((gap/tau, drift, mismatch/tau[0]))
    return curve, rate


def _mad(values, axis=None):
    center = np.median(values, axis=axis, keepdims=True)
    return 1.4826*np.median(np.abs(values-center), axis=axis)


def _fit(state, dt, times, curves, rates):
    n_curve = curves.size
    y = np.r_[curves.ravel(), rates]
    w = np.ones_like(y)

    def unpack(p):
        fast, separation = np.exp(p)
        return np.array([fast, fast + separation])

    def profile(p):
        a, b = _design(unpack(p), state, dt, times)
        a = np.concatenate((a.reshape(-1, 9), b))
        coef = np.linalg.lstsq(w[:, None]*a, w*y, rcond=None)[0]
        return np.einsum('ij,j->i', a, coef)-y, coef

    bounds = (np.log([.025, .025]), np.log([12., 70.]))

    def optimize(p):
        return least_squares(lambda v: w*profile(v)[0], p, bounds=bounds,
                             max_nfev=70, ftol=1e-9)

    # Gross observation spikes should not steer initialization of the time constants.
    center = np.median(curves, axis=0)
    scale = np.maximum(_mad(curves, axis=0), 1e-6)
    w[:n_curve] = ((np.abs(curves-center) <= 5*scale)/scale).ravel()
    w[n_curve:] = 1./max(_mad(rates), 1e-6)
    starts = np.log([[.2, 1.5], [.55, 2.8], [1.2, 5.], [.15, 8.]])
    p = min(map(optimize, starts), key=lambda fit: fit.cost).x
    for _ in range(8):
        residual, _ = profile(p)
        shape = residual[:n_curve].reshape(curves.shape)
        scale = np.maximum(_mad(shape, axis=0), max(.35*_mad(shape), 1.e-7))
        sigma = np.r_[np.broadcast_to(scale, curves.shape).ravel(),
                      np.full(rates.size, max(_mad(residual[n_curve:]), 1.e-7))]
        w = 1./np.hypot(sigma, residual/2.38)
        p = optimize(p).x
    return unpack(p), profile(p)[1]


def solve_alpha_m97_c01_batch_06(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = (np.asarray(dataset[k]) for k in ('past_dt', 'eval_times'))
    state = _prepare(dataset)
    n = len(train['lab_id'])
    curves = train['strain_curves_obs'] - train['strain_0'][:, None]
    tau, coef = _fit(tuple(a[:n] for a in state), dt, times, curves, train['instant_rate_obs'])
    curve, rate = _design(tau, tuple(a[n:] for a in state), dt, times)
    return {'estimated_taus': tau, 'hereditary_weights': coef[:7],
            'regime_mode_weights': coef[7:],
            'predicted_strain': test['strain_0'][:, None] + curve @ coef,
            'initial_rate_pred': rate @ coef}
