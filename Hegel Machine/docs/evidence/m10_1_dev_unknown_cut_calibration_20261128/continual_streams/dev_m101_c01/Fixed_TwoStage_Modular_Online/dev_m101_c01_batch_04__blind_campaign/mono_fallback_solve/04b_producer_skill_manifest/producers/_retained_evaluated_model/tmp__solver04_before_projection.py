from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _least_squares(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(panel, x):
    u, v = panel['u'], panel['v']
    keep = np.ones(len(u), dtype=bool)
    for _ in range(5):
        w = np.array([_least_squares(_affine(u[keep, j:j+1]), v[keep, j])
                      for j in range(u.shape[1])]).T
        residual = np.linalg.norm(w[0] + u * w[1] - v, axis=1)
        keep = residual < max(0.001, 4 * np.median(residual))
    return w[0] + x * w[1]


def _even_operator(panel, x):
    u, v = panel['u'], panel['v2']

    def design(q, k):
        return np.column_stack((_affine(q), np.sqrt(1 + k * q**2) - 1))

    def residual(k):
        h = design(u, k)
        return (h @ _least_squares(h, v) - v).ravel()

    k = least_squares(residual, [0.2], bounds=(0.001, 8)).x
    return design(x, k) @ _least_squares(design(u, k), v)


def _kernel_inputs(study, x):
    panel = study['panel_b']
    b = _least_squares(_affine(panel['u']), panel['v'])
    u = _affine(_calibrate(study['panel_a'], x)) @ b
    d = _even_operator(study['panel_c'], u)
    quadratic = np.column_stack([u[:, i] * u[:, j]
                                 for i, j in combinations_with_replacement(range(u.shape[1]), 2)])
    interactions = (u[:, :, None] * d[:, None, :]).reshape(len(u), -1)
    f = np.column_stack((_affine(u), quadratic, d, interactions))
    return f @ f.T, cdist(u, u, metric='sqeuclidean')


def _predict(k, train, test, observed, ridge=0.03):
    gram = k[np.ix_(train, train)] + ridge**2 * np.eye(len(train))
    return k[np.ix_(test, train)] @ solve(gram, observed, assume_a='pos')


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, q = train['x'], study['test']['x']
    i, j = np.arange(len(x)), np.arange(len(x), len(x) + len(q))
    base, distance = _kernel_inputs(study, np.vstack((x, q)))
    short = base + 0.3 * np.exp(-0.5 * distance)
    long = base + np.exp(-0.125 * distance)
    y_hat = _predict(short, i, j, train['y'])
    state_pred = (_predict(short, i, j, train['traj']) + _predict(long, i, j, train['traj'])) / 2
    final = state_pred[:, -1]
    diagnostics = np.array([y_hat.mean(), y_hat.std(), final.mean(), final.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics}
