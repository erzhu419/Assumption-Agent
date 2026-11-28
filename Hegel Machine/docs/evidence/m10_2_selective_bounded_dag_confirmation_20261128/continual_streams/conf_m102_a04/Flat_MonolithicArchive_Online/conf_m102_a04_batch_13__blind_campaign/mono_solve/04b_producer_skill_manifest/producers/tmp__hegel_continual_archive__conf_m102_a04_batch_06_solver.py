"""Calibrated product–cosine response with a three-mode trajectory model."""
import numpy as np
from scipy.optimize import least_squares


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-10)[0]


def _with_bias(a):
    return np.column_stack((np.ones(len(a)), a))


def _calibrate(panel, raw):
    u, v = np.asarray(panel['u']), np.asarray(panel['v'])
    k = u.shape[1]
    j = (np.arange(k) + 1) % k

    def terms(a):
        b = a[..., j]
        return np.stack((np.ones_like(a), a, b, a*b, a**3), axis=-1)

    f = terms(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(10):
        w = np.stack([_ls(f[keep, i], v[keep, i]) for i in range(k)])
        residual = np.linalg.norm(np.sum(f*w, axis=-1)-v, axis=1)
        refined = residual <= max(5*np.median(residual), 0.0008)
        if np.array_equal(refined, keep) or refined.sum() <= f.shape[-1]:
            break
        keep = refined

    rows = np.arange(k)
    linear = np.diag(w[:, 1])
    linear[rows, j] = w[:, 2]
    initial = np.linalg.solve(linear, (raw - w[:, 0]).T).T

    def jac(a):
        out = np.diag(w[:, 1] + w[:, 3]*a[j] + 3*w[:, 4]*a*a)
        out[rows, j] = w[:, 2] + w[:, 3]*a
        return out

    def invert(row, start):
        def residual(a):
            return np.sum(terms(a)*w, axis=-1) - row
        return least_squares(residual, start, jac=jac, gtol=1e-12).x

    return np.stack([invert(row, start) for row, start in zip(raw, initial)])


def _operators(panel, z):
    u = np.asarray(panel['u'])

    def products(a):
        return _with_bias(a[:, [0, 0, 1, 2]] * a[:, [1, 3, 2, 3]])

    def cosine(a, q):
        return _with_bias(np.cos(q*a))

    def residual(q):
        f = cosine(u, q[0])
        return (f @ _ls(f, panel['v2']) - panel['v2']).ravel()

    q = least_squares(residual, [2.4], bounds=(1.5, 3.3), gtol=1e-12).x[0]
    g = products(z) @ _ls(products(u), panel['v1'])
    h = cosine(z, q) @ _ls(cosine(u, q), panel['v2'])
    return g, h


def _cross(a, b):
    return (a[:, :, None] * b[:, None, :]).reshape(len(a), -1)


def _design(study, raw):
    z = _calibrate(study['panel_a'], raw)
    g, h = _operators(study['panel_c'], z)
    contraction = np.sum(g * h, axis=1)
    common = _with_bias(np.column_stack((g, h, _cross(z, g), contraction)))
    return common, np.column_stack((common, g[:, 0] * h[:, 1]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    fy, fs = _design(study, np.vstack((train['x'], study['test']['x'])))
    traj = np.asarray(train['traj'])
    basis = np.linalg.svd(traj, full_matrices=False)[2][:3]
    y = fy[n:] @ _ls(fy[:n], train['y'])
    state = (fs[n:] @ _ls(fs[:n], traj @ basis.T)) @ basis
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
