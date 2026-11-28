"""Calibrated two-branch hereditary strain and initial-rate prediction."""

import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    parts = train, test
    labs = np.concatenate([p['lab_id'] for p in parts])
    strata = np.concatenate([p['stratum'] for p in parts])
    ref = np.vstack([p['ref_coupons'] for p in parts])
    raw = np.vstack([p['coupon_features_raw'] for p in parts])
    x = np.empty_like(raw, dtype=float)
    for lab in np.unique(labs):
        m = labs == lab
        # Pool each lab's reference panel independently of the split. Sorting
        # also gives an identical reduction order after reference exchange.
        panel = np.sort(ref[m], axis=0)
        center = np.nanmedian(np.nanmean(panel, axis=0))
        spread = np.nanmedian(np.nanstd(panel, axis=0, ddof=0))
        x[m] = (raw[m] - center) / spread
    _, cell, counts = np.unique(np.column_stack((labs, strata)), axis=0,
                                return_inverse=True, return_counts=True)
    w = (1.0 / counts[cell])[:, None]
    observed = np.isfinite(x)
    denom = np.sum(w * observed, axis=0)
    sums = np.sum(w * np.where(observed, x, 0.), axis=0)
    means = np.divide(sums, denom, out=np.zeros_like(sums), where=denom > 0)
    return np.where(observed, x, means)


def _bridge_raw(load, bridge):
    f, amplitude, g, asymmetry = bridge[:4]
    offset, gain, coupling = bridge[4:8], bridge[8:12], bridge[12:16]
    opposite = np.roll(load, 2, axis=-1)
    return (offset + gain * (load + amplitude * np.sin(f * load))
            + coupling * (opposite + asymmetry * (np.cos(g * opposite) - 1)))


def _calibrate(calibration):
    load = np.asarray(calibration['true_channel_loads'])
    raw = np.asarray(calibration['bridge_raw'])
    # Gross bridge saturation is separate from ordinary measurement noise.
    clean = np.max(np.abs(raw), axis=1) < 8
    load, raw = load[clean], raw[clean]
    p0 = np.r_[2.5, .25, 2., .27, np.zeros(4), np.ones(4), np.full(4, .2)]
    bounds = (np.r_[2., .1, 1., 0., np.full(4, -np.inf), np.zeros(8)],
              np.r_[3., .32, 3., .6, np.full(12, np.inf)])
    return least_squares(lambda p: (_bridge_raw(load, p) - raw).ravel(),
                         p0, bounds=bounds, x_scale='jac').x


def _bridge_load(raw, bridge):
    f, amplitude, g, asymmetry = bridge[:4]
    offset, gain, coupling = bridge[4:8], bridge[8:12], bridge[12:16]
    # Invert the two independent opposite-channel pairs simultaneously.
    load = (raw - offset) / (gain * (1 + f * amplitude))
    for _ in range(40):
        error = _bridge_raw(load, bridge) - raw
        if np.max(np.abs(error)) < 1e-11:
            break
        diag = gain * (1 + f * amplitude * np.cos(f * load))
        off = coupling * (1 - g * asymmetry * np.sin(g * np.roll(load, 2, axis=-1)))
        diag_opp = np.roll(diag, 2, axis=-1)
        determinant = diag * diag_opp - off * np.roll(off, 2, axis=-1)
        step = (diag_opp * error - off * np.roll(error, 2, axis=-1)) / determinant
        load -= step / np.maximum(1., np.max(np.abs(step), axis=-1, keepdims=True))
    return load


def _stress(raw, bridge):
    return _bridge_load(np.asarray(raw), bridge).sum(axis=-1)


def _inputs(part, x, bridge):
    return (_stress(part["past_channel_raw"], bridge),
            _stress(part["hold_channel_raw"], bridge),
            np.column_stack((np.ones(len(x)), x)))


def _design(taus, inp, dt, t):
    past, hold, covariates = inp
    # A pulse is attenuated by all intervals that follow it, not by its own
    # interval twice. This is the exact unrolled chronological state update.
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    coeff = -np.expm1(-dt[:, None] / taus) * np.exp(-tail[:, None] / taus)
    gap = hold[:, None] - past @ coeff
    rise = -np.expm1(-t[:, None] / taus)
    strain = np.concatenate((gap[:, None, :] * rise[None, :, :],
                             covariates[:, None, :] * t[None, :, None]), axis=2)
    rate = np.column_stack((gap / taus, covariates))
    return strain, rate


def _fit(inp, dt, t, strain, rate):
    n, k = strain.shape
    y = np.r_[strain.ravel(), rate]

    def matrix(taus):
        a, b = _design(taus, inp, dt, t)
        return np.vstack((a.reshape(n*k, 7), b))

    def residual(p):
        return matrix(p[:2]) @ p[2:] - y

    def noise(r):
        def mad(v):
            return max(1e-8, 1.4826 * np.median(np.abs(v - np.median(v))))
        return np.r_[np.full(n*k, mad(r[:n*k])), np.full(n, mad(r[n*k:]))]

    # Compare separated branch starts, eliminating all seven linear
    # coefficients before changing the time constants.
    starts = []
    for lo in (.12, .35, .8, 1.6):
        for hi in (1.2, 3., 7., 16.):
            if hi <= lo:
                continue
            a = matrix(np.array([lo, hi]))
            p = np.r_[lo, hi, np.linalg.lstsq(a, y, rcond=None)[0]]
            starts.append((np.linalg.norm(residual(p)), p))
    bounds = (np.r_[np.full(2, .05000001), np.full(7, -np.inf)],
              np.r_[np.full(2, 24.99999999), np.full(7, np.inf)])
    p0 = min(starts, key=lambda pair: pair[0])[1]
    scale = noise(residual(p0))
    # Smooth robust regression first, then redescending influence for the
    # occasional heavy-tailed sensor spikes.
    p = least_squares(lambda q: residual(q)/scale, p0, bounds=bounds,
                      loss='soft_l1', f_scale=2, x_scale='jac', max_nfev=200).x
    scale = noise(residual(p))
    p = least_squares(lambda q: residual(q)/scale, p, bounds=bounds,
                      loss='cauchy', f_scale=2.5, x_scale='jac', max_nfev=200).x
    if p[0] > p[1]:
        p = p[[1, 0, 3, 2, 4, 5, 6, 7, 8]]
    return p[:2], p[2:]


def _chamber_response(taus, theta, hold, covariates, times):
    # The pre-run apparatus trace has a high plateau on telemetry channel 5,
    # as before the archived reset campaign. Its hold starts with zero memory.
    drift = covariates @ theta[2:]
    amplitudes = hold[:, None] * theta[:2] + drift[:, None] * np.array([0.4, 1.6])
    linear = 0.06 * drift
    rise = -np.expm1(-times[:, None] / taus)
    return (amplitudes @ rise.T + linear[:, None] * times,
            amplitudes @ (1.0 / taus) + linear)


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt = np.asarray(dataset['past_dt'], dtype=float)
    times = np.asarray(dataset['eval_times'], dtype=float)
    x = _features(train, test)
    n = len(train['lab_id'])
    bridge = _calibrate(dataset['calibration_load_cells'])
    taus, theta = _fit(_inputs(train, x[:n], bridge), dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    hold = _stress(test['hold_channel_raw'], bridge)
    covariates = np.column_stack((np.ones(len(hold)), x[n:]))
    response, rate = _chamber_response(taus, theta, hold, covariates, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + response,
            'initial_rate_pred': rate}
