from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def _quadratic(x):
    return np.column_stack([np.ones(len(x)), x, *(x[:, i] * x[:, j]
        for i, j in combinations_with_replacement(range(x.shape[1]), 2))])


def _readout(x, xt, y):
    p, pt = _quadratic(x), _quadratic(xt)
    a = np.column_stack([np.ones(len(x)), x])
    at = np.column_stack([np.ones(len(xt)), xt])
    d = np.sum((x[:, None] - x[None, :])**2, axis=2)
    dt = np.sum((xt[:, None] - x[None, :])**2, axis=2)
    k, kt = p @ p.T, pt @ p.T
    pred = np.zeros(len(xt))
    for weight, length, scale, noise, dot in (
        (0.65, 1.5, 10., .002, True),
        (0.35, 1.2, 10., .005, False),
    ):
        g = np.exp(-d / (2 * length**2))
        gt = np.exp(-dt / (2 * length**2))
        if dot:
            g *= a @ a.T
            gt *= at @ a.T
        factor = cho_factor(k + scale * g + noise * np.eye(len(x)))
        pred += weight * ((kt + scale * gt) @ cho_solve(factor, y))
    return pred


def _state(x, xt, z):
    mode = np.linalg.svd(z, full_matrices=False)[2][0]
    amplitude = z @ mode
    p, pt = _quadratic(x), _quadratic(xt)
    d = (x[:, None] - x[None, :])**2
    dt = (xt[:, None] - x[None, :])**2
    k, kt = p @ p.T, pt @ p.T
    eye = np.eye(len(x))

    def kernel(d, base, h):
        r = np.sqrt(5 * (d @ np.exp(h[:4])))
        return base + np.exp(h[4]) * (1 + r + r*r/3) * np.exp(-r)

    def objective(h):
        factor = cho_factor(kernel(d, k, h) + np.exp(h[5]) * eye)
        alpha = cho_solve(factor, amplitude)
        return (0.5 * amplitude @ alpha + np.log(factor[0].diagonal()).sum()
                + .2 * np.sum((h[:4] - h[:4].mean())**2))

    h = minimize(objective, [-2., -2., -2., -2., -1., -10.],
                 bounds=[(-7, 3)]*4 + [(-9, 4), (-13, -8)],
                 method='L-BFGS-B').x
    factor = cho_factor(kernel(d, k, h) + np.exp(h[5]) * eye)
    amp_test = kernel(dt, kt, h) @ cho_solve(factor, amplitude)
    return amp_test[:, None] * mode


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = np.asarray(train['x']), np.asarray(study['test']['x'])
    y = _readout(x, xt, np.asarray(train['y']))
    state = _state(x, xt, np.asarray(train['traj']))
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
