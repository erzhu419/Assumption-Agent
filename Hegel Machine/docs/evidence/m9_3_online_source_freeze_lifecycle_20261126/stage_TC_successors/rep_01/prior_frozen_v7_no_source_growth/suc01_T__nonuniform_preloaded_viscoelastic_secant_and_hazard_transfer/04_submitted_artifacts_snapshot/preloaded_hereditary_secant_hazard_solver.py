"""Two-branch hereditary transfer from nonuniform, preloaded histories."""
import numpy as np
from scipy.optimize import least_squares


def _branch(tau, rows, dt, times):
    memory = np.array(rows['initial_memory_states'], dtype=float, copy=True)
    past = np.asarray(rows['past_stress'])
    for j, step in enumerate(dt):
        memory += -np.expm1(-step / tau) * (past[:, j, None] - memory)
    gap = np.asarray(rows['stress_hold_0'])[:, None] - memory
    return gap[:, None, :] * -np.expm1(-times[:, None] / tau), gap / tau


def _fit_physics(rows, dt, times):
    target = np.column_stack((np.asarray(rows['strain_curves_obs']) - np.asarray(rows['strain_0'])[:, None], rows['instant_rate_obs']))
    scale = np.r_[np.full(times.size, .014), .017]
    t = np.r_[times, 1.] / scale

    def decode(p):
        return np.array([p[0], p[0] + (9. - p[0]) * p[1]])

    def profile(p):
        curves, rate = _branch(decode(p), rows, dt, times)
        centered = (target - np.column_stack((curves @ p[2:], rate @ p[2:]))) / scale
        drift = np.median(centered / t, axis=1)
        for _ in range(18):
            residual = centered - drift[:, None] * t
            weights = 1. / (1. + residual**2)
            drift = np.sum(weights * centered * t, axis=1) / np.sum(weights * t**2, axis=1)
        return centered - drift[:, None] * t, drift

    best = None
    for lo, hi in ((.48, 3.4), (.27, 1.4), (.95, 6.), (2.2, 7.5)):
        start = [lo, (hi - lo) / (9. - lo), .7, .5]
        fit = least_squares(lambda p: profile(p)[0].ravel(), start,
                            bounds=([.2, 1e-6, -np.inf, -np.inf], [9. - 1e-6, 1., np.inf, np.inf]),
                            loss='cauchy', x_scale='jac', max_nfev=100)
        if best is None or fit.cost < best.cost:
            best = fit
    return decode(best.x), best.x[2:], profile(best.x)[1]


def _coupons(train, test):
    labs = np.r_[train['lab_id'], test['lab_id']]
    refs = np.vstack((train['ref_coupons'], test['ref_coupons']))
    x = np.vstack((train['coupon_features_raw'], test['coupon_features_raw'])).astype(float)
    for lab in np.unique(labs):
        sel = labs == lab
        x[sel] = (x[sel] - np.nanmean(refs[sel])) / np.nanstd(refs[sel])
    n = len(train['lab_id'])
    cells = np.column_stack((train['lab_id'], train['stratum']))
    _, ids, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
    w = 1. / counts[ids]
    observed = np.isfinite(x)
    fill = np.sum(np.where(observed[:n], x[:n], 0.) * w[:, None], axis=0) / np.sum(observed[:n] * w[:, None], axis=0)
    return np.where(observed, x, fill), observed, w


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _predict_drift(train, test, fitted, train_rate, test_rate):
    x, observed, w = _coupons(train, test)
    n = len(fitted)
    design = np.column_stack((np.ones(len(x)), x))
    complete = observed[:n].all(axis=1)

    def regress(sel):
        root = np.sqrt(w[sel])
        return np.linalg.lstsq(design[:n][sel] * root[:, None], fitted[sel] * root, rcond=None)[0]

    common = regress(complete)
    prior = design @ common
    # Use labeled complete specimens to remove small residual calibration errors
    # after panel normalization; keep the common calibration for an unseen lab.
    tlab = np.asarray(train['lab_id'])
    lab = np.r_[tlab, test['lab_id']]
    for j in np.unique(tlab):
        sel = (tlab == j) & complete
        if np.count_nonzero(sel) >= 2 * design.shape[1]:
            prior[lab == j] = design[lab == j] @ regress(sel)

    rate_sd = max(_mad(train_rate - fitted), .002)
    coupon_sd = max(_mad((fitted - prior[:n])[complete]), .0015)
    test_prior = prior[n:]
    missing = ~observed[n:]
    # Missing features make the balanced-fill prediction less informative.
    missing_scale = np.std(x[:n], axis=0) * common[1:]
    coupon_variance = coupon_sd**2 + missing @ (missing_scale**2)
    rate_weight = coupon_variance / (coupon_variance + rate_sd**2)
    return test_prior + rate_weight * (test_rate - test_prior)


def _crossings(times, initial, curves, critical):
    time = np.r_[0., times]
    strain = np.column_stack((initial, curves))
    reached = strain >= np.asarray(critical)[:, None]
    first = np.argmax(reached, axis=1)
    out = np.full(len(strain), time[-1])
    out[reached[:, 0]] = 0.
    rows = np.flatnonzero(first > 0)
    j = first[rows]
    low = strain[rows, j - 1]
    fraction = (np.asarray(critical)[rows] - low) / (strain[rows, j] - low)
    out[rows] = time[j - 1] + fraction * (time[j] - time[j - 1])
    return out


def solve_preloaded_hereditary_transfer(dataset: dict) -> dict:
    """Estimate shared relaxation times, then transfer strain and terminal readouts."""
    train, test = dataset['train'], dataset['test']
    dt, times = np.asarray(dataset['past_dt']), np.asarray(dataset['eval_times'])
    tau, amplitudes, fitted = _fit_physics(train, dt, times)
    _, train_rate = _branch(tau, train, dt, times)
    curves, test_rate = _branch(tau, test, dt, times)
    drift = _predict_drift(train, test, fitted,
                           np.asarray(train['instant_rate_obs']) - train_rate @ amplitudes,
                           np.asarray(test['instant_rate_obs']) - test_rate @ amplitudes)
    initial = np.asarray(test['strain_0'])
    prediction = initial[:, None] + curves @ amplitudes + drift[:, None] * times
    return {
        'estimated_taus': tau,
        'predicted_strain': prediction,
        'secant_compliance': (prediction[:, -1] - initial) / np.maximum(np.abs(test['stress_hold_0']), .5),
        'hazard_crossing_times': _crossings(times, initial, prediction, test['critical_strain']),
    }
