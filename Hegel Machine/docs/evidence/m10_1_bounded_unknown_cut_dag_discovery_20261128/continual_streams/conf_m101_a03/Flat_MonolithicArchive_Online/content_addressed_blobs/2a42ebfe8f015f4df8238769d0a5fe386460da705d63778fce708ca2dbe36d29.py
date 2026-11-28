"""Panel calibration, a bounded scalar readout, and low-rank trajectory prediction."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(f, y):
    return np.linalg.lstsq(f, y, rcond=None)[0]


def _coordinates(panel, x):
    u, v = panel['u'], panel['v']
    keep = np.arange(len(u))
    for step in range(5):
        coeff = np.array([_fit(_affine(u[keep, j:j+1]), v[keep, j])
                          for j in range(u.shape[1])])
        if step < 4:
            err = np.linalg.norm(coeff[:, 0] + u * coeff[:, 1] - v, axis=1)
            keep = np.argsort(err)[:round(.9375 * len(u))]
    return (x - coeff[:, 0]) / coeff[:, 1]


def _operator(panel, z, compact=False):
    u, v = panel['u'], panel['v']
    if compact:
        def design(w, p):
            return _affine(w * (p[1] - w*w) / (1 + np.exp(p[0])*w*w))
        start = [-.24, .05]
    else:
        def design(w, p):
            return np.column_stack((_affine(w), w / (1 + np.exp(p)*w*w)))
        start = [-.24, 0., 0., 0.]

    def residual(p):
        f = design(u, p)
        return (f @ _fit(f, v) - v).ravel()

    p = least_squares(residual, start, bounds=(-5, 5), max_nfev=100).x
    return design(z, p) @ _fit(design(u, p), v)


def _scalar(b, y):
    n = len(y)
    w = b / b[:n].std(axis=0)
    linear = _affine(w)
    pairs = w[:, [0, 1]] * w[:, [2, 3]]
    base = np.column_stack((linear, pairs))
    # Subtract the linear part of tanh, which is already in the base.
    penalty = np.diag(np.sqrt(np.r_[np.full(5, 1e-8), .0003, .0003, .003]))
    target = np.r_[y, np.zeros(penalty.shape[0])]

    def design(rows, p):
        h = linear[rows] @ p
        return np.column_stack((base[rows], np.tanh(h) - h))

    def coefficients(f):
        return _fit(np.vstack((f, penalty)), target)

    def residual(p):
        f = design(slice(n), p)
        c = coefficients(f)
        return np.r_[f @ c - y, penalty @ c, .005*p]

    k = _fit(linear[:n], y)
    starts = [k*.3, k, k*2, *(np.eye(5)[1:] * .7)]
    p = min((least_squares(residual, init, bounds=(-6, 6), max_nfev=90)
             for init in starts), key=lambda sol: sol.cost).x
    return design(slice(n, None), p) @ coefficients(design(slice(n), p))


def _trajectory(features, observed):
    n = len(observed)
    center = observed.mean(axis=0)
    centered = observed - center
    modes = np.linalg.svd(centered, full_matrices=False)[2][:2]
    weights = _fit(features[:n], centered @ modes.T)
    return center + features[n:] @ weights @ modes


def solve_batch(study: dict) -> dict:
    train = study['train']
    z = _coordinates(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    b = _operator(study['panel_b'], z)
    compact = _operator(study['panel_b'], z, compact=True)
    y = .5*(_scalar(b, train['y']) + _scalar(compact, train['y']))
    state = .5*_trajectory(np.column_stack((_affine(z), b)), train['traj'])
    state += .5*_trajectory(_affine(compact), train['traj'])
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
