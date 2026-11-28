"""Predict scalar and trajectory readouts from the panel B operator."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:, i]*x[:, j]
                            for i, j in combinations_with_replacement(range(x.shape[1]), 2))))


def _panel_b(panel, x):
    u, v = panel['u'], panel['v']

    def design(z, frequencies):
        return np.column_stack((_affine(z), np.tanh(z*frequencies)))

    def residual(frequencies):
        a = design(u, frequencies)
        return (a @ _fit(a, v) - v).ravel()

    frequencies = least_squares(residual, [1.4]*4, bounds=(.25, 5.5)).x
    return design(x, frequencies) @ _fit(design(u, frequencies), v)


def _relaxation(traj, t, initial):
    def basis(rates):
        return -np.expm1(-t[:, None]*rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _fit(b, traj.T) - traj.T).ravel()

    rates = least_squares(residual, initial, bounds=([.7, .05], [4., .65])).x
    modes = np.linalg.qr(basis(rates), mode='reduced')[0]
    return traj @ modes, modes


def _forecast(d2, d2_test, p, pt, response, width, ridge):
    k = np.exp(-width*d2)
    k.flat[::len(k)+1] += ridge
    factor = cho_factor(k)
    q, y = cho_solve(factor, p), cho_solve(factor, response)
    w = _fit(p.T @ q, p.T @ y)
    return pt @ w + np.exp(-width*d2_test) @ (y - q @ w)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = train['x'], study['test']['x']
    n = len(x)
    trend = _quadratic(_panel_b(study['panel_b'], np.vstack((x, xt))))
    p, pt = trend[:n], trend[n:]
    d2 = cdist(x, x, 'sqeuclidean')
    d2_test = cdist(xt, x, 'sqeuclidean')
    modes, basis = _relaxation(train['traj'], study['eval_times'], [1.8, .41])
    y_hat = _forecast(d2, d2_test, p, pt, train['y'], .075, .0003)
    state_pred = _forecast(d2, d2_test, p, pt, modes, .035, .0003) @ basis.T
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
