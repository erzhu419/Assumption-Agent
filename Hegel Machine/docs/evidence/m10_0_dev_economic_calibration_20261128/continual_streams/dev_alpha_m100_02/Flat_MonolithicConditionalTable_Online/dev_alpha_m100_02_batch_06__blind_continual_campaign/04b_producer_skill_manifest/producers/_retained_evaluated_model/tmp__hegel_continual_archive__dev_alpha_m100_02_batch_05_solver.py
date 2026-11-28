"""Fit the load bridge and material law for the equilibrated chamber."""

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


_OPPOSITE = np.array([2, 3, 0, 1])


def _bridge(calibration):
    """Fit the shared trigonometric bridge law from axis and mixed sweeps."""
    raw = np.asarray(calibration['bridge_raw'])
    loads = np.asarray(calibration['true_channel_loads'])
    sweeps = calibration['single_channel_sweeps']
    axis_raw = np.asarray(sweeps['cmd_stress'])
    # The gross readout jumps are separated from the clean sweeps by >10 units.
    clean = np.max(np.abs(raw), axis=1) < 5.0
    axis_clean = np.abs(axis_raw) < 5.0
    labs = np.asarray(sweeps['lab_id'])[axis_clean]
    axis_x = np.asarray(sweeps['true_stress'])[axis_clean]
    axis_y = axis_raw[axis_clean]
    x, y = loads[clean], raw[clean]
    w = x[:, _OPPOSITE]

    def axis_design(freq):
        return np.column_stack((np.ones_like(axis_x), axis_x, np.sin(freq * axis_x)))

    def profile_axis(freq, coefficients=False):
        a = axis_design(freq[0])
        b = np.stack([np.linalg.lstsq(a[labs == j], axis_y[labs == j], rcond=None)[0]
                      for j in range(4)])
        return b if coefficients else (a * b[labs]).sum(axis=1) - axis_y

    f = least_squares(profile_axis, [2.5], bounds=(1.0, 4.0)).x
    axis = profile_axis(f, coefficients=True)
    offset, gain, sine_amplitude = axis.T
    residual = y - offset - gain*x - sine_amplitude*np.sin(f[0]*x)

    def profile_cross(freq, coefficients=False):
        a = np.column_stack(((gain*w).ravel(), (gain*(1-np.cos(freq[0]*w))).ravel()))
        b = np.linalg.lstsq(a, residual.ravel(), rcond=None)[0]
        return b if coefficients else a @ b - residual.ravel()

    g = least_squares(profile_cross, [2.0], bounds=(0.5, 4.0)).x
    cross, cosine = profile_cross(g, coefficients=True)
    return offset, gain, sine_amplitude / gain, f[0], cross, cosine, g[0]


def _loads(raw, calibration):
    offset, gain, sine, f, cross, cosine, g = calibration
    y = (np.asarray(raw) - offset) / gain
    # Newton inversion treats the two opposite-channel pairs independently.
    x = y.copy()
    for _ in range(20):
        w = x[..., _OPPOSITE]
        error = x + sine*np.sin(f*x) + cross*w + cosine*(1-np.cos(g*w)) - y
        diag = 1 + sine*f*np.cos(f*x)
        off = cross + cosine*g*np.sin(g*w)
        determinant = diag*diag[..., _OPPOSITE] - off*off[..., _OPPOSITE]
        step = (diag[..., _OPPOSITE]*error - off*error[..., _OPPOSITE]) / determinant
        x -= step
        if np.max(np.abs(step)) < 1e-11:
            break
    return x


def _prepare(part, x, calibration):
    past = _loads(part['past_channel_raw'], calibration).sum(axis=-1)
    hold = _loads(part['hold_channel_raw'], calibration).sum(axis=-1)
    return past, hold, np.column_stack((np.ones(len(x)), x))


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


def solve_batch(dataset: dict) -> dict:
    train, test = dataset['train'], dataset['test']
    times, dt = np.asarray(dataset['eval_times']), np.asarray(dataset['past_dt'])
    x_train, x_test = _features(train, test)
    calibration = _bridge(dataset['calibration_load_cells'])
    tau, theta = _fit(_prepare(train, x_train, calibration), dt, times,
                      train['strain_curves_obs'] - train['strain_0'][:, None],
                      train['instant_rate_obs'])

    # The latest equilibration trace matches the two archived baseline regimes:
    # neither chamber-reset channel is elevated. Keep the full stress history.
    curves, rate = _design(np.log(tau), _prepare(test, x_test, calibration), dt, times)
    return {'estimated_taus': tau, 'hereditary_weights': theta,
            'predicted_strain': test['strain_0'][:, None] + curves @ theta,
            'initial_rate_pred': rate @ theta}
