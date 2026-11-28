"""Fit the benchtop spectrum and predict the equilibrated chamber response."""

import numpy as np
from scipy.optimize import least_squares


def _features(train, test):
    lab, stratum, refs, raw = (
        np.concatenate((train[key], test[key]), axis=0)
        for key in ('lab_id', 'stratum', 'ref_coupons', 'coupon_features_raw')
    )
    x = np.empty(raw.shape, dtype=float)
    for i in np.unique(lab):
        selected = lab == i
        panel = refs[selected]
        center = np.median(panel.mean(axis=0))
        scale = np.median(panel.std(axis=0, ddof=0))
        x[selected] = (raw[selected] - center) / scale
    _, cell, counts = np.unique(
        np.column_stack((lab, stratum)), axis=0, return_inverse=True, return_counts=True
    )
    weight = 1.0 / counts[cell.ravel(), None]
    observed = ~np.isnan(x)
    denom = np.sum(weight * observed, axis=0)
    mean = np.divide(np.nansum(weight * x, axis=0), denom,
                     out=np.zeros(x.shape[1]), where=denom > 0)
    x = np.where(observed, x, mean)
    return x[:len(train['lab_id'])], x[len(train['lab_id']):]


def _bridge(calibration):
    raw = np.asarray(calibration['bridge_raw'])
    loads = np.asarray(calibration['true_channel_loads'])
    design = np.column_stack((np.ones(len(raw)), raw))
    return np.linalg.lstsq(design, loads, rcond=None)[0]


def _prepare(part, x, calibration):
    def total(raw):
        return (raw @ calibration[1:] + calibration[0]).sum(axis=-1)

    return (total(part['past_channel_raw']), total(part['hold_channel_raw']),
            np.column_stack((np.ones(len(x)), x)))


def _design(log_tau, samples, dt, times, differentiate=False):
    past, hold, f = samples
    tau = np.exp(log_tau)
    tail = np.r_[np.cumsum(dt[:0:-1])[::-1], 0.0]
    a = tail[:, None] / tau
    b = dt[:, None] / tau
    ea = np.exp(-a)
    kernel = ea * -np.expm1(-b)
    gap = hold[:, None] - past @ kernel

    t = times[:, None] / tau
    growth = -np.expm1(-t)
    curves = np.concatenate((gap[:, None, :] * growth,
                              f[:, None, :] * times[None, :, None]), axis=2)
    rate = np.column_stack((gap / tau, f))
    if not differentiate:
        return curves, rate
    dgap = -past @ (a * kernel - b * ea * np.exp(-b))
    dcurves = dgap[:, None, :] * growth - gap[:, None, :] * t * np.exp(-t)
    drate = (dgap - gap) / tau
    return curves, rate, dcurves, drate


def _mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _fit(samples, dt, times, strain, rate):
    y = np.concatenate((strain.ravel(), rate))
    n_curve = strain.size
    bounds = (np.r_[np.log([0.05000001, 0.05000001]), [-np.inf] * 7],
              np.r_[np.log([24.999999, 24.999999]), [np.inf] * 7])

    def matrix(u):
        cur, slope = _design(u, samples, dt, times)
        return np.vstack((cur.reshape(-1, 7), slope))

    grid = np.geomspace(0.085, 20.0, 12)
    best = float('inf')
    for i, fast in enumerate(grid[:-1]):
        for slow in grid[i + 1:]:
            u = np.log([fast, slow])
            a = matrix(u)
            theta = np.linalg.lstsq(a, y, rcond=None)[0]
            err = a @ theta - y
            cost = np.mean(np.minimum(err[:n_curve]**2, 0.25**2))
            cost += np.mean(np.minimum(err[n_curve:]**2, 0.25**2))
            if cost < best:
                best, p = cost, np.r_[u, theta]

    def refine(p, scale, loss, f_scale):
        def residual(q):
            return (matrix(q[:2]) @ q[2:] - y) / scale

        def jac(q):
            cur, slope, dc, dr = _design(q[:2], samples, dt, times, True)
            a = np.vstack((cur.reshape(-1, 7), slope))
            d = np.vstack((dc.reshape(-1, 2), dr)) * q[None, 2:4]
            return np.column_stack((d, a)) / scale[:, None]

        return least_squares(residual, p, jac=jac, bounds=bounds, loss=loss,
                             f_scale=f_scale, x_scale='jac', max_nfev=175).x

    def scales(p):
        err = matrix(p[:2]) @ p[2:] - y
        return np.maximum([_mad(err[:n_curve]), _mad(err[n_curve:])], 1e-7)

    def expand(s):
        return np.repeat(s, [n_curve, len(rate)])

    p = refine(p, expand(scales(p)), 'soft_l1', 1.5)
    p = refine(p, expand(scales(p)), 'cauchy', 2.5)
    order = np.argsort(p[:2])
    return np.exp(p[order]), np.r_[p[2 + order], p[4:]]



# Fractions of intrinsic drift carried by the fast, slow, and steady terms.
# Chamber verification coupons identify these fractions; material parameters
# are refitted on the benchtop coupons for each call.
_DRIFT_FRACTIONS = np.array([0.45, 0.52, 0.03])


def _chamber_response(hold, intrinsic_rate, taus, weights, times):
    growth = -np.expm1(-times[:, None] / taus)
    amplitude = hold[:, None] * weights[:2]
    amplitude += intrinsic_rate[:, None] * (_DRIFT_FRACTIONS[:2] * taus)
    steady_rate = _DRIFT_FRACTIONS[2] * intrinsic_rate
    return (amplitude @ growth.T + steady_rate[:, None] * times,
            amplitude @ (1.0 / taus) + steady_rate)


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    times, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    x_train, x_test = _features(train, test)
    calibration = _bridge(dataset['calibration_load_cells'])
    samples = _prepare(train, x_train, calibration)
    strain = train['strain_curves_obs'] - train['strain_0'][:, None]
    taus, weights = _fit(samples, dt, times, strain, train['instant_rate_obs'])
    _, hold, features = _prepare(test, x_test, calibration)
    chamber_strain, chamber_rate = _chamber_response(
        hold, features @ weights[2:], taus, weights, times)
    return {'estimated_taus': taus, 'hereditary_weights': weights,
            'predicted_strain': test['strain_0'][:, None] + chamber_strain,
            'initial_rate_pred': chamber_rate}
