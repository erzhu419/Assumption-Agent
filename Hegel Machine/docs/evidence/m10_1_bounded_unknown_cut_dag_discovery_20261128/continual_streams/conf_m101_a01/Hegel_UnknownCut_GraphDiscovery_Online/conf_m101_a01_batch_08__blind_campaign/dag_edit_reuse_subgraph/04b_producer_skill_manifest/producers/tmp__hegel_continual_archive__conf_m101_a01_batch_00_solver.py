from itertools import combinations_with_replacement

import numpy as np


def _quadratic(x):
    pairs = combinations_with_replacement(range(x.shape[1]), 2)
    return np.column_stack([np.ones(len(x)), x] + [x[:, i] * x[:, j] for i, j in pairs])


def _matern52(u, v):
    r = (np.sqrt(5.0) / 2.4) * np.linalg.norm(u[:, None, :] - v[None, :, :], axis=2)
    return (1.0 + r + r * r / 3.0) * np.exp(-r)


def _predict(a, b, kernel, cross_kernel, target, regularization):
    # Fit the quadratic trend in the kernel metric, then predict its residual.
    w = np.linalg.solve(kernel + regularization * np.eye(len(a)), np.column_stack((a, target)))
    wa, wy = w[:, :a.shape[1]], w[:, a.shape[1]:]
    trend = np.linalg.solve(a.T @ wa, a.T @ wy)
    return b @ trend + cross_kernel @ (wy - wa @ trend)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, q = train['x'], study['test']['x']
    a, b = _quadratic(x), _quadratic(q)
    kernel, cross_kernel = _matern52(x, x), _matern52(q, x)
    y = _predict(a, b, kernel, cross_kernel, train['y'], 0.001)[:, 0]
    state = _predict(a, b, kernel, cross_kernel, train['traj'], 0.003)
    final = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), final.mean(), final.std()])}
