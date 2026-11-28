"""Recalibrate the two panels and fit independent scalar and trajectory readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import minimize_scalar, root


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _profile(design, target, grid, width):
    def score(w):
        f = design(w)
        return np.mean((f @ _ls(f, target) - target)**2)
    w0 = grid[np.argmin([score(w) for w in grid])]
    w = minimize_scalar(score, bounds=(w0-width, w0+width), method='bounded').x
    return w, _ls(design(w), target)


def _inverse(panel, x, gain, offset):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    size = np.max(np.abs(v), axis=1)
    mid = np.median(size)
    clean = size < mid + max(1.5, 8*np.median(np.abs(size-mid)))
    if clean.sum() >= 12:
        u, v = u[clean], v[clean]
    def design(q, j, w):
        return np.c_[np.ones(len(q)), q, np.sin(w*q[:, (j+2)%4]), np.sin(w*q[:, j])]
    calibrations = [_profile(lambda w: design(u, j, w), v[:, j], np.linspace(.8, 5., 75), .07) for j in range(4)]
    def forward(q):
        return np.array([c[0] + q @ c[1:5] + c[5]*np.sin(w*q[(j+2)%4]) + c[6]*np.sin(w*q[j]) for j, (w, c) in enumerate(calibrations)])
    out = np.empty_like(x, dtype=float)
    safe_gain = np.where(np.abs(gain) > 1e-6, gain, 1.)
    for i, row in enumerate(x):
        fun = lambda q: forward(q) - row
        start = (row-offset)/safe_gain
        sol = root(fun, start, tol=1e-10).x
        if not np.all(np.isfinite(sol)) or np.linalg.norm(fun(sol)) > 1e-7 or np.max(np.abs(sol)) > 1.4:
            alt = root(fun, row/1.2, tol=1e-10).x
            if np.all(np.isfinite(alt)) and (not np.all(np.isfinite(sol)) or np.linalg.norm(fun(alt)) < np.linalg.norm(fun(sol)) or (np.linalg.norm(fun(alt)) <= 1e-7 and np.linalg.norm(alt) < np.linalg.norm(sol))):
                sol = alt
        out[i] = sol if np.all(np.isfinite(sol)) else start
    return out


def _even_panel(panel, q):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def design(x, w):
        return np.c_[np.ones(len(x)), np.cos(w*x)]
    w, coef = _profile(lambda w: design(u, w), v, np.linspace(.5, 6., 111), .06)
    return design(q, w) @ coef


def _sparse_readout(f, target, n, terms=6):
    mu = f[:n].mean(0)
    scale = np.maximum(f[:n].std(0), 1e-6)
    norm = (f-mu)/scale
    a = norm[:n]
    target = np.asarray(target)
    center = target.mean(0)
    y = target-center
    residual = y
    columns = []
    for _ in range(min(terms, a.shape[1])):
        correlation = np.linalg.norm((a.T @ residual).reshape(a.shape[1], -1), axis=1)
        correlation[columns] = -1.
        columns.append(int(np.argmax(correlation)))
        coef = _ls(a[:, columns], y)
        residual = y-a[:, columns] @ coef
    return norm[n:, columns] @ coef + center


def run_node(inputs: dict) -> dict:
    study, upstream = inputs['__input__'], inputs['node_00']
    train = study['train']
    n = len(train['x'])
    x = np.vstack((train['x'], study['test']['x'])).astype(float)
    q = _inverse(study['panel_a'], x, np.asarray(upstream['gain']), np.asarray(upstream['offset']))
    b = _even_panel(study['panel_b'], q)
    scalar = np.column_stack([b] + [b[:, i]*b[:, j] for i, j in combinations_with_replacement(range(b.shape[1]), 2)])
    y = _sparse_readout(scalar, train['y'], n)
    z = _sparse_readout(np.c_[q, b], train['traj'], n)
    last = z[:, -1]
    return {'y_hat': y, 'state_pred': z, 'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
