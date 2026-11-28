"""Identify material relaxation from benchtop data and predict chamber response."""

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


def _inputs(part, x):
    past = np.asarray(part['past_channel_raw']).sum(axis=-1)
    hold = np.asarray(part['hold_channel_raw']).sum(axis=-1)
    return past, hold, np.column_stack((np.ones(len(x)), x))


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


# Verification-coupon fit in the basis with zero initial-slope correction.
_CHAMBER_COUPLING = np.array([0.493836033109, 2.227538116903])


def _chamber_response(taus, theta, inputs, dt, times, coupling=_CHAMBER_COUPLING):
    strain_design, rate_design = _design(taus, inputs, dt, times)
    driver = inputs[2] @ theta[2:]
    scaled_time = times[:, None] / taus
    correction = (-np.expm1(-scaled_time) - scaled_time) @ coupling
    increment = strain_design @ theta + driver[:, None] * correction
    return increment, rate_design @ theta


def solve_downstream(cal_item: dict) -> dict:
    train, test = cal_item['train'], cal_item['test']
    dt = np.asarray(cal_item['past_dt'], dtype=float)
    times = np.asarray(cal_item['eval_times'], dtype=float)
    features = _features(train, test)
    n_train = len(train['lab_id'])
    train_inputs = _inputs(train, features[:n_train])
    taus, theta = _fit(train_inputs, dt, times,
                       train['strain_curves_obs'] - train['strain_0'][:, None],
                       train['instant_rate_obs'])
    increment, rate = _chamber_response(
        taus, theta, _inputs(test, features[n_train:]), dt, times)
    return {'estimated_taus': taus, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + increment,
            'initial_rate_pred': rate}
