"""Recover panel operators and regress their coupled temporal response."""
import numpy as np
from scipy.optimize import least_squares


def _ols(f, y):
    return np.linalg.lstsq(f, y, rcond=1e-10)[0]


def _one(x):
    return np.column_stack((np.ones(len(x)), x))


def _align(panel):
    u, v = panel['u'], panel['v']
    i, j = np.triu_indices(len(u), 1)
    shifts, scales = [], []
    for a, b in zip(u.T, v.T):
        da = a[i] - a[j]
        k = abs(da) > 1e-8
        scale = np.median((b[i[k]] - b[j[k]]) / da[k])
        shift = np.median(b - scale*a)
        f = _one(a)
        for _ in range(5):
            d = abs(b - shift - scale*a)
            keep = d <= max(5*np.median(d), 2e-4)
            shift, scale = _ols(f[keep], b[keep])
        shifts.append(shift)
        scales.append(scale)
    return np.asarray(shifts), np.asarray(scales)


def _products(z):
    return z[:, [0, 0, 1, 2]] * z[:, [1, 3, 2, 3]]


def _curvature(z, q):
    sq = z*z
    return _one(np.c_[z, sq, sq/(1 + q*sq)])


def _features(study, x):
    shift, scale = _align(study['panel_a'])
    z = (x-shift)/scale
    c = study['panel_c']
    u = c['u']
    a = _one(_products(z)) @ _ols(_one(_products(u)), c['v1'])

    def residual(p):
        f = _curvature(u, p[0])
        return (f @ _ols(f, c['v2']) - c['v2']).ravel()

    q = least_squares(residual, [0.3], bounds=(0.02, 3.0)).x[0]
    b = _curvature(z, q) @ _ols(_curvature(u, q), c['v2'])
    return a, b


def _temporal(t, traj):
    def basis(r):
        return -np.expm1(-t[:, None]*r)

    def residual(r):
        f = basis(r)
        return (f @ _ols(f, traj.T) - traj.T).ravel()

    r = least_squares(residual, [0.42, 1.82], bounds=([0.015, 0.75], [0.75, 8])).x
    return basis(r)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    a, b = _features(study, np.r_[train['x'], study['test']['x']])
    interactions = (a[:, :, None] * b[:, None, :]).reshape(len(a), 16)
    g = _one(np.c_[a, b, (a*b).sum(axis=1)])
    f = _one(np.c_[a, b, interactions])
    temporal = _temporal(np.asarray(study['eval_times']), train['traj'])
    y = g[n:] @ _ols(g[:n], train['y'])
    state = (f[n:] @ _ols(f[:n], _ols(temporal, train['traj'].T).T)) @ temporal.T
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
