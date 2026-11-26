"""Fit and transfer a two-branch hereditary creep law."""
import numpy as np
from scipy.optimize import least_squares


def _cell_weights(lab, stratum):
    _, cell, count = np.unique(np.column_stack((lab, stratum)), axis=0,
                               return_inverse=True, return_counts=True)
    return len(lab) / (len(count) * count[cell.ravel()])


def _coupons(train, test):
    lab = np.r_[train['lab_id'], test['lab_id']]
    ref = np.concatenate((train['ref_coupons'], test['ref_coupons']))
    raw = np.concatenate((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for label in np.unique(lab):
        mask = lab == label
        panel = ref[mask]
        panel = panel[:, np.isfinite(panel).sum(axis=0) >= 2]
        offset = np.median(np.nanmean(panel, axis=0))
        gain = np.median(np.nanstd(panel, axis=0))
        raw[mask] = (raw[mask] - offset) / gain
    n = len(train['lab_id'])
    w = _cell_weights(train['lab_id'], train['stratum'])
    valid = np.isfinite(raw)
    observed = valid[:n]
    mass = w @ observed
    total = w @ np.where(observed, raw[:n], 0.)
    fill = np.divide(total, mass, out=np.zeros_like(total), where=mass > 0)
    features = np.column_stack((np.ones(len(raw)), np.where(valid, raw, fill)))
    complete = valid.all(axis=1)
    return features[:n], features[n:], complete[:n], complete[n:], w


def _branches(split, dt, times, tau):
    # Closed form of the chronological exponential recurrence, including preload.
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.]
    kernel = -np.expm1(-dt[:, None] / tau) * np.exp(-tail[:, None] / tau)
    end = (split['past_stress'] @ kernel +
           split['initial_memory_states'] * np.exp(-sum(dt) / tau))
    gap = split['stress_hold_0'][:, None] - end
    return gap[:, None, :] * -np.expm1(-times[:, None] / tau), gap / tau


def _fit(train, dt, t, x, complete, weight):
    y = train['strain_curves_obs'] - train['strain_0'][:, None]
    rate = train['instant_rate_obs']
    w = np.sqrt(weight)
    curve_scale, rate_scale = .018, .014
    contrast_scale = np.hypot(curve_scale, rate_scale * t)

    def unpack(q):
        fast = q[0]
        return np.array([fast, fast + q[1] * (9. - fast)])

    def residual(q):
        tau = unpack(q)
        curve, slope = _branches(train, dt, t, tau)
        branch_curve, branch_rate = curve @ q[2:4], slope @ q[2:4]
        drift = x @ q[4:]
        a = (branch_curve[complete] + drift[complete, None] * t - y[complete]) / curve_scale
        b = (branch_rate[complete] + drift[complete] - rate[complete]) / rate_scale
        m = ~complete
        c = (branch_curve[m] + (rate[m] - branch_rate[m])[:, None] * t - y[m]) / contrast_scale
        return np.r_[(a * w[complete, None]).ravel(), b * w[complete], (c * w[m, None]).ravel()]

    bounds = ([.2, 1e-6, -np.inf, -np.inf] + [-np.inf] * x.shape[1],
              [9. - 1e-5, 1., np.inf, np.inf] + [np.inf] * x.shape[1])
    start = np.r_[.55, (3.5 - .55)/(9. - .55), .7, .5, np.zeros(x.shape[1])]
    ans = least_squares(residual, start, bounds=bounds, loss='cauchy', f_scale=2.,
                        max_nfev=160)
    return unpack(ans.x), ans.x[2:4], ans.x[4:]


def _correct_lab_drift(train, test, dt, times, tau, amplitude, beta,
                       x, xt, complete):
    # Refine the reference-panel offset and gain using late training curves.
    # An unseen lab keeps its reference-only estimate.
    curve, _ = _branches(train, dt, times, tau)
    late = slice(len(times) // 2 - 1, None)
    residual_curve = train['strain_curves_obs'] - train['strain_0'][:, None] - curve @ amplitude
    measured = np.median(residual_curve[:, late] / times[late], axis=1)
    centered = x[:, 1:] @ beta[1:]
    centered_test = xt[:, 1:] @ beta[1:]
    drift = xt @ beta
    for lab in np.unique(train['lab_id'][complete]):
        mask = (train['lab_id'] == lab) & complete
        target = test['lab_id'] == lab
        if mask.sum() < 2 or not np.any(target):
            continue
        u = centered[mask]
        innovation = measured[mask] - beta[0] - u
        offset, gain_error = least_squares(
            lambda q: (q[0] + q[1] * u - innovation) / .005,
            [0., 0.], loss='soft_l1', f_scale=1.5).x
        drift[target] += offset + gain_error * centered_test[target]
    return drift


def _crossings(times, initial, curves, critical):
    out = np.full(len(initial), times[-1], dtype=float)
    out[initial >= critical] = 0.
    hit = curves >= critical[:, None]
    rows = np.flatnonzero((initial < critical) & hit.any(axis=1))
    j = hit[rows].argmax(axis=1)
    y0 = np.column_stack((initial, curves))[rows, j]
    t0 = np.r_[0., times][j]
    out[rows] = t0 + (times[j] - t0) * (critical[rows] - y0) / (curves[rows, j] - y0)
    return out


def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    x, xt, complete, complete_test, w = _coupons(train, test)
    tau, amplitude, drift_coef = _fit(train, dt, times, x, complete, w)
    curve, rate = _branches(test, dt, times, tau)
    rate_drift = test['instant_rate_obs'] - rate @ amplitude
    coupon_drift = _correct_lab_drift(train, test, dt, times, tau, amplitude,
                                     drift_coef, x, xt, complete)
    drift = np.where(complete_test, .75 * coupon_drift + .25 * rate_drift, rate_drift)
    initial = test['strain_0']
    prediction = initial[:, None] + curve @ amplitude + drift[:, None] * times
    return {'estimated_taus': tau, 'predicted_strain': prediction,
            'secant_compliance': (prediction[:, -1] - initial) / np.maximum(np.abs(test['stress_hold_0']), .5),
            'hazard_crossing_times': _crossings(times, initial, prediction, test['critical_strain'])}
