from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares
from scipy.spatial.distance import cdist


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _coordinates(study, x):
    panel = study['panel_a']
    design = _affine(panel['u'])
    mask = np.ones(len(design), dtype=bool)
    for _ in range(5):
        a = _fit(design[mask], panel['v'][mask])
        err = np.linalg.norm(panel['v'] - design @ a, axis=1)
        mask = err <= max(3 * np.median(err), 1e-12)
    panel = study['panel_b']
    b = _fit(_affine(panel['u']), panel['v'])
    return _affine(_affine(x) @ a) @ b


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:, j] * x[:, k]
        for j, k in combinations_with_replacement(range(x.shape[1]), 2))))


def _temporal_basis(t, trajectories):
    def basis(p):
        return np.column_stack((np.exp(-p[0] * t) * np.sin(p[1] * t),
                                -np.expm1(-p[2] * t)))

    def residual(p):
        b = basis(p)
        return (b @ _fit(b, trajectories.T) - trajectories.T).ravel()

    p = least_squares(residual, [0.2, 1.1, 0.55],
                      bounds=([0, 0.5, 0.08], [1, 2, 2])).x
    return basis(p)


def _kernel(left, right, scale):
    d = cdist(left / scale, right / scale, 'sqeuclidean') / left.shape[1]
    return _quadratic(left) @ _quadratic(right).T + 0.3 * np.exp(-1.3 * d)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    r = _coordinates(study, np.vstack((train['x'], study['test']['x'])))
    x, xt = r[:n], r[n:]
    scale = x.std(axis=0)
    b = _temporal_basis(study['eval_times'], train['traj'])
    target = np.column_stack((train['y'], _fit(b, train['traj'].T).T))
    gram = _kernel(x, x, scale) + np.eye(n) * 1e-5
    pred = _kernel(xt, x, scale) @ cho_solve(cho_factor(gram), target)
    # Blend a smoother residual for the trajectory amplitudes, where
    # the shorter-range kernel is less reliable beyond the training cloud.
    q, qt = _quadratic(x), _quadratic(xt)
    ds = cdist(x / scale, x / scale, 'sqeuclidean') / x.shape[1]
    dt = cdist(xt / scale, x / scale, 'sqeuclidean') / x.shape[1]
    smooth = q @ q.T + 0.035 * np.exp(-0.45 * ds) + np.eye(n) * 1e-5
    amplitudes = (qt @ q.T + 0.035 * np.exp(-0.45 * dt)) @ cho_solve(cho_factor(smooth), target[:, 1:])
    pred[:, 1:] = 0.65 * pred[:, 1:] + 0.35 * amplitudes
    y, state = pred[:, 0], pred[:, 1:] @ b.T
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
