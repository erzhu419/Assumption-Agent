"""Invert the reference calibration, then fit scalar and relaxation readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _least(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(u):
    return np.column_stack((np.ones(len(u)), u))


def _calibration_design(u, j):
    k = (j + 1) % u.shape[1]
    a, b = u[:, j], u[:, k]
    return np.column_stack((np.ones(len(u)), a, b, a*b, a**3))


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    designs = [_calibration_design(u, j) for j in range(u.shape[1])]
    distance = np.linalg.norm(v-u, axis=1)
    keep = distance < max(5*np.median(distance), 1e-8)
    for _ in range(4):
        w = np.column_stack([_least(g[keep], v[keep, j]) for j, g in enumerate(designs)])
        err = np.linalg.norm(np.column_stack([g @ w[:, j] for j, g in enumerate(designs)])-v, axis=1)
        keep = err < max(6*np.median(err), 1e-8)
    return w


def _inverse(x, w):
    j = np.arange(x.shape[1]); k = np.roll(j, -1)
    linear = np.diag(w[1]); linear[j, k] = w[2]
    u = np.linalg.solve(linear, (x-w[0]).T).T
    jac = np.zeros((len(u), len(j), len(j)))
    for _ in range(40):
        nxt = u[:, k]
        r = w[0] + w[1]*u + w[2]*nxt + w[3]*u*nxt + w[4]*u**3 - x
        if np.max(np.abs(r), initial=0.) < 1e-11:
            break
        jac[:, j, j] = w[1] + w[3]*nxt + 3*w[4]*u**2
        jac[:, j, k] = w[2] + w[3]*u
        du = np.linalg.solve(jac, r[..., None])[..., 0]
        u -= du/np.maximum(1., np.max(np.abs(du), axis=1, keepdims=True))
    return u


def _scalar(g, gt, y):
    pairs = list(combinations_with_replacement(range(g.shape[1]), 2))
    def design(x):
        return np.column_stack((_affine(x), *(x[:, i]*x[:, j] for i,j in pairs)))
    a, b = design(g), design(gt)
    p = _least(a, y)
    keep = np.abs(p) > .04
    keep[:1+g.shape[1]] = True
    return b[:, keep] @ _least(a[:, keep], y)


def _trajectory(g, gt, times, z):
    a, b = _affine(g), _affine(gt)
    # In the calibration-corrected coordinates, the amplitudes are affine.
    # Profile the rates against the spatially explained trajectories.
    project = _least(a, z)
    target = a @ project
    def basis(p):
        return -np.expm1(-times[:, None] * np.exp(p))
    def residual(p):
        f = basis(p)
        return (f @ _least(f, target.T) - target.T).ravel()
    fit = least_squares(residual, np.log([.4, 1.8]), bounds=(np.log([.05, .9]), np.log([.89, 8.])), gtol=1e-11)
    f = basis(fit.x)
    return (b @ _least(f, project.T).T) @ f.T


def solve_batch(study: dict) -> dict:
    tr = study['train']
    x, xt = np.asarray(tr['x'], dtype=float), np.asarray(study['test']['x'], dtype=float)
    w = _calibrate(study['panel_a'])
    u, ut = np.split(_inverse(np.concatenate((x, xt)), w), [len(x)])
    panel = study['panel_b']
    op = _least(_affine(panel['u']), panel['v'])
    g, gt = _affine(u) @ op, _affine(ut) @ op
    y = _scalar(g, gt, tr['y'])
    z = _trajectory(g, gt, np.asarray(study['eval_times']), tr['traj'])
    return {'y_hat': y, 'state_pred': z,
            'diagnostics_vec': np.array([y.mean(), y.std(), z[:, -1].mean(), z[:, -1].std()])}
