"""Calibrated panel interaction model with two relaxation modes."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=1e-9)[0]


def _affine(x):
    return np.c_[np.ones(len(x)), x]


def _cyclic(x):
    return x * np.roll(x, -1, axis=1)


def _coordinates(study):
    panel = study['panel_a']
    a, v = _affine(panel['u']), panel['v']
    m = _fit(a, v)
    for _ in range(12):
        w = np.minimum(1., .07 / np.maximum(np.linalg.norm(a @ m - v, axis=1), 1e-12))
        m = _fit(a * w[:, None], v * w[:, None])
    good = np.linalg.norm(a @ m - v, axis=1) < .01
    m = _fit(a[good], v[good])
    x = np.vstack([study['train']['x'], study['test']['x']])
    return np.linalg.solve(m[1:].T, (x - m[0]).T).T


def _views(study, z):
    c = study['panel_c']
    u = c['u']
    p = _affine(_cyclic(z)) @ _fit(_affine(_cyclic(u)), c['v1'])
    def f(x, k):
        return np.c_[_affine(x), np.log1p(k * x * x)]
    def loss(k):
        g = f(u, k)
        return np.mean((g @ _fit(g, c['v2']) - c['v2'])**2)
    k = minimize_scalar(loss, bounds=(.05, 1.5), method='bounded').x
    q = f(z, k) @ _fit(f(u, k), c['v2'])
    return p, q


def _temporal(t, s):
    def basis(r):
        return -np.expm1(-t[:, None] * r)
    def residual(r):
        h = basis(r)
        return (h @ _fit(h, s.T) - s.T).ravel()
    h = basis(least_squares(residual, [.4, 2.], bounds=(.001, 12.)).x)
    return h, _fit(h, s.T).T


def solve_batch(study):
    tr = study['train']
    n = len(tr['x'])
    z = _coordinates(study)
    h, modes = _temporal(study['eval_times'], tr['traj'])
    base = np.c_[_affine(z), _cyclic(z), z*z]
    read = _fit(np.c_[base[:n], modes], tr['y'])
    p, q = _views(study, z)
    shared = np.c_[base, q]
    slow = np.c_[shared, p[:, 0] * q[:, 1]]
    fast = np.c_[shared, np.sum(p*q, axis=1)]

    # Estimate both spatial modes jointly from the measured trajectories and response.
    gs, gf = slow[:n], fast[:n]
    nt = h.shape[0]
    temporal = np.concatenate([
        gs[:, None, :] * h[None, :, 0, None],
        gf[:, None, :] * h[None, :, 1, None],
    ], axis=2).reshape(n * nt, gs.shape[1] + gf.shape[1])
    design = np.r_[temporal, np.c_[read[-2] * gs, read[-1] * gf]]
    target = np.r_[tr['traj'].ravel(), tr['y'] - base[:n] @ read[:-2]]
    penalty = .005 * np.linalg.norm(design, axis=0)
    w = _fit(np.r_[design, np.diag(penalty)], np.r_[target, np.zeros(len(penalty))])
    d = gs.shape[1]
    pred = np.c_[slow[n:] @ w[:d], fast[n:] @ w[d:]]
    y = base[n:] @ read[:-2] + pred @ read[-2:]
    state = pred @ h.T
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
