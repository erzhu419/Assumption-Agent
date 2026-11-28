import numpy as np
from scipy.optimize import least_squares


def _a_features(u):
    n = np.roll(u, -1, axis=1)
    return np.stack((np.ones_like(u), u, n, u**3, u*n), axis=2)


def _fit_a(panel):
    u, v = panel['u'], panel['v']
    d = _a_features(u)
    weights = np.ones(len(u))
    for _ in range(24):
        c = np.stack([np.linalg.lstsq(d[:, j]*weights[:, None], v[:, j]*weights, rcond=None)[0] for j in range(4)])
        r = np.linalg.norm(np.sum(d*c, axis=2)-v, axis=1)
        scale = max(1e-6, 2*np.median(r))
        weights = 1/np.hypot(1, r/scale)
    return c


def _forward_a(u, c):
    return np.sum(_a_features(u)*c, axis=2)


def _inverse_a(v, c):
    j = np.arange(4)
    nxt = (j+1)%4
    linear = np.diag(c[:, 1])
    linear[j, nxt] = c[:, 2]
    u = np.linalg.solve(linear, (v-c[:, 0]).T).T
    for _ in range(16):
        r = _forward_a(u, c)-v
        if np.max(np.abs(r)) < 1e-10:
            break
        jac = np.broadcast_to(linear, (len(u), 4, 4)).copy()
        jac[:, j, j] += 3*c[:, 3]*u*u+c[:, 4]*u[:, nxt]
        jac[:, j, nxt] += c[:, 4]*u
        u -= np.linalg.solve(jac, r[..., None])[..., 0]
    for i in np.flatnonzero(np.max(np.abs(_forward_a(u, c)-v), axis=1) > 1e-8):
        u[i] = least_squares(lambda q: _forward_a(q[None], c)[0]-v[i], u[i]).x
    return u


def _fit_b(panel):
    u, v = panel['u'], panel['v']
    return np.linalg.lstsq(np.column_stack((np.ones(len(u)), u)), v, rcond=None)[0]
