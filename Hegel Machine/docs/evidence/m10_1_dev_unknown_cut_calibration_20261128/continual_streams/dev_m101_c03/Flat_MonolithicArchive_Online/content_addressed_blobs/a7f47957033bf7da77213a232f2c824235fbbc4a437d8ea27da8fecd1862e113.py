"""Panel-calibrated endpoint and two-mode trajectory prediction."""

import numpy as np
from scipy.optimize import least_squares, root


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _decode(panel, readings):
    u, v = panel['u'], panel['v']

    def terms(z):
        neighbor = np.roll(z, -1, axis=-1)
        return np.stack((np.ones_like(z), z, neighbor, z * neighbor, z**3), axis=-1)

    design = terms(u)
    keep = np.ones(len(u), dtype=bool)
    # Outliers occur on shared rows across channels. Trim by joint residual.
    for _ in range(4):
        coef = np.array([_fit(design[keep, j], v[keep, j]) for j in range(u.shape[1])])
        residual = np.linalg.norm(v - np.sum(design * coef, axis=-1), axis=1)
        keep = residual <= 6 * np.median(residual)
    coef = np.array([_fit(design[keep, j], v[keep, j]) for j in range(u.shape[1])])

    def decode(row):
        def residual(z):
            return np.sum(terms(z) * coef, axis=-1) - row
        return root(residual, row, tol=1e-10).x

    return np.array([decode(row) for row in readings])


def _branch(u, v, z, first, orthogonal=False):
    def features(x, p):
        return p[0] * x + np.tanh(p[1] * x) if first else x + p[0] * x**2

    def operator(p):
        x = features(u, p)
        if not orthogonal:
            return _fit(_affine(x), v)
        a, b = x.mean(0), v.mean(0)
        xc = x - a
        left, values, right = np.linalg.svd(xc.T @ (v - b))
        matrix = (left @ right) * (values.sum() / np.sum(xc**2))
        return np.vstack((b - a @ matrix, matrix))

    def residual(p):
        return (_affine(features(u, p)) @ operator(p) - v).ravel()

    start, bounds = ([0.45, 2.2], ([0.001, 0.1], [3, 5])) if first else ([0.45], (0, 2))
    p = least_squares(residual, start, bounds=bounds).x
    return _affine(features(z, p)) @ operator(p)


def _features(study, x, orthogonal=False):
    z = _decode(study['panel_a'], x)
    p = study['panel_c']
    a = _branch(p['u'], p['v1'], z, True, orthogonal)
    b = _branch(p['u'], p['v2'], z, False, orthogonal)
    return a, b


def _predict_modes(a, b, n, times, observed):
    dot = np.sum(a * b, axis=1)
    slow = np.column_stack((b, a[:, 0] * b[:, 1]))
    fast = np.column_stack((a, dot))

    def temporal(p):
        ts, tf = -np.expm1(-p[:2, None] * times)
        return ts + p[2] * tf, tf

    def profile(p):
        ts, tf = temporal(p)
        design = np.column_stack(((slow[:n, None, :] * ts[None, :, None]).reshape(observed.size, -1),
                                  (fast[:n, None, :] * tf[None, :, None]).reshape(observed.size, -1)))
        coef = _fit(design, observed.ravel())
        return design @ coef - observed.ravel(), coef

    p = least_squares(lambda p: profile(p)[0], [0.4, 1.8, 0.12],
                      bounds=([0.1, 1.0, -0.4], [0.85, 3.0, 0.6])).x
    _, coef = profile(p)
    cs, cf = np.split(coef, [slow.shape[1]])
    ts, tf = temporal(p)
    return np.outer(slow[n:] @ cs, ts) + np.outer(fast[n:] @ cf, tf)


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.vstack((train['x'], study['test']['x']))
    n = len(train['x'])
    a, b = _features(study, x)
    design = _affine(np.column_stack((a, b, np.sum(a * b, axis=1))))
    y = design[n:] @ _fit(design[:n], train['y'])
    a, b = _features(study, x, orthogonal=True)
    state = _predict_modes(a, b, n, study['eval_times'], train['traj'])
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
