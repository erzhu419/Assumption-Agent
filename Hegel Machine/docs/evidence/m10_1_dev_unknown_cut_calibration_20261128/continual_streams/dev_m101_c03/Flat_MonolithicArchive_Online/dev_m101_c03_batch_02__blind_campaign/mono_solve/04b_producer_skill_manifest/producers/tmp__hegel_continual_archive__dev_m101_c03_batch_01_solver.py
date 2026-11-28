"""Predict the endpoint and two-mode trajectory from calibrated panel operators."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    coef = []
    for x, y in zip(u.T, v.T):
        dx = x[i] - x[j]
        ok = dx != 0
        slope = np.median((y[i] - y[j])[ok] / dx[ok])
        offset = np.median(y - slope * x)
        error = np.abs(y - offset - slope * x)
        keep = error <= 6 * np.median(error) + 1e-8
        coef.append(_fit(_affine(x[keep]), y[keep]))
    return np.asarray(coef).T


def _cross_basis(z):
    return _affine(z[:, [0, 0, 1, 2]] * z[:, [1, 3, 2, 3]])


def _even_basis(z, p):
    return _affine(z + p[0] * (np.sqrt(1 + p[1] * z**2) - 1))


def _design(study, inputs):
    offset, gain = _calibrate(study['panel_a'])
    z = (inputs - offset) / gain
    panel = study['panel_c']
    u, v1, v2 = panel['u'], panel['v1'], panel['v2']
    a = _cross_basis(z) @ _fit(_cross_basis(u), v1)

    def residual(p):
        f = _even_basis(u, p)
        return (f @ _fit(f, v2) - v2).ravel()

    p = least_squares(residual, [1.0, 1.0], bounds=([0, 0.01], [10, 10])).x
    b = _even_basis(z, p) @ _fit(_even_basis(u, p), v2)
    return np.column_stack((_affine(z), a, b, a[:, 0] * b[:, 1], np.sum(a * b, axis=1)))


def _time_basis(times, rates):
    return -np.expm1(-times[:, None] * rates)


def _kinetics(x, obs, times):
    sampled = x @ _fit(x, obs)

    def residual(rates):
        basis = _time_basis(times, rates)
        return (_fit(basis, sampled.T).T @ basis.T - sampled).ravel()

    rates = least_squares(residual, [0.4, 1.8], bounds=([0.05, 1.0], [1.0, 4.0])).x
    basis = _time_basis(times, rates)
    return basis, _fit(basis, obs.T).T


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    design = _design(study, np.vstack((train['x'], study['test']['x'])))
    x, xt = design[:n], design[n:]
    basis, amplitudes = _kinetics(x, train['traj'], np.asarray(study['eval_times']))
    state = xt @ _fit(x, amplitudes) @ basis.T
    # The endpoint uses the two branches and their dot product.
    endpoint_cols = np.r_[0, 5:13, 14]
    y = xt[:, endpoint_cols] @ _fit(x[:, endpoint_cols], train['y'])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([
        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()
    ])}
