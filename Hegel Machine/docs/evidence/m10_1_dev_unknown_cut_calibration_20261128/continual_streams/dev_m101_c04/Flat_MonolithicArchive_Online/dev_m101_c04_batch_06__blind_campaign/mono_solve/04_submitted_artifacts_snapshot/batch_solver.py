"""Calibrated quadratic dynamics and rational scalar readout."""

import itertools
import numpy as np
from scipy.optimize import least_squares

def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration_features(x, j):
    p, q = x[:, j], x[:, (j + 1) % 4]
    return _affine(np.column_stack((p, q, p * q, p**3)))


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    design = [_calibration_features(u, j) for j in range(4)]
    rng = np.random.default_rng(37)
    best = np.inf
    for _ in range(48):
        rows = rng.choice(len(u), 8, replace=False)
        fit = np.column_stack([a @ _ls(a[rows], v[rows, j]) for j, a in enumerate(design)])
        errors = np.linalg.norm(v - fit, axis=1)
        score = np.median(errors)
        if score < best:
            best = score
            keep = errors <= 5 * score
    return np.array([_ls(a[keep], v[keep, j]) for j, a in enumerate(design)])


def _unwarp(x, coeff):
    offset, gain, neighbor, cross, cubic = coeff.T
    j = np.arange(4)
    k = (j + 1) % 4
    linear = np.diag(gain)
    linear[j, k] = neighbor
    u = np.linalg.solve(linear, (x - offset).T).T
    jac = np.zeros((len(x), 4, 4))
    for _ in range(20):
        q = u[:, k]
        residual = offset + gain*u + neighbor*q + cross*u*q + cubic*u**3 - x
        if np.max(np.abs(residual)) < 1e-12:
            break
        jac[:, j, j] = gain + cross*q + 3*cubic*u**2
        jac[:, j, k] = neighbor + cross*u
        delta = np.linalg.solve(jac, residual[..., None])[..., 0]
        u -= delta / np.maximum(1, np.linalg.norm(delta, axis=1, keepdims=True))
    return u


def _coordinates(study):
    coeff = _calibrate(study['panel_a'])
    b = study['panel_b']
    mixing = _ls(_affine(b['u']), b['v'])
    return tuple(_affine(_unwarp(study[k]['x'], coeff)) @ mixing for k in ('train', 'test'))


_PAIRS = np.array(list(itertools.combinations_with_replacement(range(4), 2)))


def _features(z):
    return _affine(z), np.prod(z[:, _PAIRS], axis=2), z[:, :2] * z[:, 2:]


def _trajectory_mode(u, q, traj):
    # Estimate the nonlinear direction independently of affine effects.
    w = _ls(q - u @ _ls(u, q), traj - u @ _ls(u, traj))
    _, _, vt = np.linalg.svd(w, full_matrices=False)
    return w @ vt[0]


def _scalar_readout(train, test, y):
    u, q, _, _ = train
    p0 = _ls(np.column_stack((u, q)), y)[1:5]
    p0 /= np.linalg.norm(p0)

    def basis(data, p):
        u, _, pairs, shared = data
        cross = pairs.sum(axis=1)
        score = u @ p
        gate = 1 / (1 + score**2)
        return np.column_stack((u, shared, cross, score * gate, cross * gate))

    def residual(p):
        a = basis(train, p)
        return a @ _ls(a, y) - y

    fit = min((least_squares(residual, np.r_[0., p0 * scale], bounds=(-6, 6),
                             max_nfev=125) for scale in (0.7, 1.2, 2.2)),
              key=lambda result: result.cost)
    a = basis(train, fit.x)
    return basis(test, fit.x) @ _ls(a, y)


def _predict(train, test, y, traj):
    u, q, pairs = _features(train)
    v, r, test_pairs = _features(test)
    w = _trajectory_mode(u, q, traj)
    shared = q @ w
    test_shared = r @ w
    state = np.column_stack((v, test_shared)) @ _ls(np.column_stack((u, shared)), traj)
    response = _scalar_readout((u, q, pairs, shared), (v, r, test_pairs, test_shared), y)
    return response, state


def solve_batch(study: dict) -> dict:
    train, test = _coordinates(study)
    y, state = _predict(train, test, study['train']['y'], study['train']['traj'])
    final = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), final.mean(), final.std()])}
