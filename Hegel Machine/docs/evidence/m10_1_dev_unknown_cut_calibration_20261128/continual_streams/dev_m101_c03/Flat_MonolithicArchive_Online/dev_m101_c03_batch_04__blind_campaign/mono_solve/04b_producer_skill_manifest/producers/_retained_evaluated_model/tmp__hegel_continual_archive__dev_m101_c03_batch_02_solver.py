"""Predict the response from calibrated panel-C interactions and two rise modes."""
import numpy as np
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _offset_column(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = panel['u'], panel['v']
    left, right = np.triu_indices(len(u), 1)
    coeff = []
    for x, y in zip(u.T, v.T):
        delta = x[left] - x[right]
        valid = delta != 0
        slope = np.median((y[left][valid] - y[right][valid]) / delta[valid])
        detrended = y - slope * x
        err = np.abs(detrended - np.median(detrended))
        keep = err <= 6 * np.median(err) + 1e-8
        coeff.append(_fit(_offset_column(x[keep]), y[keep]))
    return np.array(coeff).T


def _branch(u, v, z, activation, initial):
    """Profile the offset and scaled orthogonal mixing for each rate."""
    center = v.mean(axis=0)

    def profile(rate):
        f = activation(u, rate)
        mean = f.mean(axis=0)
        a = f - mean
        left, singular, right = np.linalg.svd(a.T @ (v - center))
        mix = (left @ right) * (singular.sum() / np.sum(a * a))
        return a @ mix + center - v, mean, mix

    rate = least_squares(lambda p: profile(p[0])[0].ravel(), [initial],
                         bounds=(1., 3.)).x[0]
    _, mean, mix = profile(rate)
    return (activation(z, rate) - mean) @ mix + center


def _features(study, x):
    offset, gain = _calibration(study['panel_a'])
    z = (x - offset) / gain
    p = study['panel_c']
    a = _branch(p['u'], p['v1'], z,
                lambda u, k: -np.sign(u) * np.expm1(-k * np.abs(u)), 1.8)
    b = _branch(p['u'], p['v2'], z, lambda u, k: np.cos(k * u), 2.4)
    return a, b, np.sum(a * b, axis=1), a[:, 0] * b[:, 1]


def _time_basis(t, rates):
    return -np.expm1(-t[:, None] * rates)


def _fit_modes(slow, fast, t, y):
    """Fit a shared slow driver with a small contribution to the fast rise."""
    def profile(params):
        bs, bf = _time_basis(t, params[:2]).T
        x = np.column_stack((
            (slow[:, None, :] * (bs + params[2] * bf)[None, :, None]).reshape(y.size, slow.shape[1]),
            (fast[:, None, :] * bf[None, :, None]).reshape(y.size, fast.shape[1]),
        ))
        coef = _fit(x, y.ravel())
        return x @ coef - y.ravel(), coef

    params = least_squares(lambda p: profile(p)[0], [0.42, 1.8, 0.1],
                           bounds=([0.08, 1., -0.5], [0.9, 3., 0.5])).x
    _, coef = profile(params)
    bs, bf = _time_basis(t, params[:2]).T
    return np.vstack((bs + params[2] * bf, bf)), np.split(coef, [slow.shape[1]])


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    a, b, dot, cross = _features(study, np.vstack((train['x'], study['test']['x'])))
    endpoint = _offset_column(np.column_stack((a, b, dot)))
    slow = np.column_stack((b, cross))
    fast = np.column_stack((a, dot))
    y = endpoint[n:] @ _fit(endpoint[:n], train['y'])
    basis, (cs, cf) = _fit_modes(slow[:n], fast[:n], study['eval_times'], train['traj'])
    state = np.column_stack((slow[n:] @ cs, fast[n:] @ cf)) @ basis
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([
        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()
    ])}
