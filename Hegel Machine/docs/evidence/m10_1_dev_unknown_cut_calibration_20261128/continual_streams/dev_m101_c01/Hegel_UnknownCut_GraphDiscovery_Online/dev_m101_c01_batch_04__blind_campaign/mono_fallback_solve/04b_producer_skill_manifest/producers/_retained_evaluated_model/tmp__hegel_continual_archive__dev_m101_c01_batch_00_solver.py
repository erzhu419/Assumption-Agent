from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _quadratic(x):
    return np.column_stack((np.ones(len(x)), x, *(
        x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2)
    )))


def _temporal_basis(times, trajectories):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)

    def residual(rates):
        u = basis(rates)
        coefficients = np.linalg.lstsq(u, trajectories.T, rcond=None)[0]
        return (u @ coefficients - trajectories.T).ravel()

    rates = least_squares(residual, [1.5, 0.22], bounds=([0.7, 0.025], [3.0, 0.65])).x
    return basis(rates)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'])
    xt = np.asarray(study['test']['x'])
    y = np.asarray(train['y'])
    z = np.asarray(train['traj'])
    t = np.asarray(study['eval_times'])
    f, ft = _quadratic(x), _quadratic(xt)
    p, pt = f @ f.T, ft @ f.T
    d, dt = cdist(x, x, 'sqeuclidean'), cdist(xt, x, 'sqeuclidean')
    eye = np.eye(len(x))
    y_hat = np.zeros(len(xt))
    for weight, length, noise in ((0.75, 2.0, 0.0001), (0.25, 1.4, 0.001)):
        k = p + np.exp(-d / (2 * length**2))
        kt = pt + np.exp(-dt / (2 * length**2))
        y_hat += weight * (kt @ solve(k + noise * eye, y, assume_a='pos'))

    def matern(r):
        q = np.sqrt(5 * r) / 3
        return (1 + q + q*q/3) * np.exp(-q)

    basis = _temporal_basis(t, z)
    z_smooth = (basis @ np.linalg.lstsq(basis, z.T, rcond=None)[0]).T
    state = (pt + 0.10 * matern(dt)) @ solve(p + 0.10 * matern(d) + 0.00003 * eye, z_smooth, assume_a='pos')
    end = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
