"""Panel-calibrated scalar and rank-two trajectory regression."""
from itertools import combinations_with_replacement

import numpy as np


def _cubic(q):
    terms = [np.prod(q[:, idx], axis=1) for d in (1, 2, 3)
             for idx in combinations_with_replacement(range(q.shape[1]), d)]
    return np.column_stack([np.ones(len(q)), *terms])


def _predict(x, targets, penalty):
    n = len(targets)
    reg = penalty * np.eye(_cubic(x[:1]).shape[1])

    def fit(k):
        w = np.sqrt(k)
        design = _cubic(np.sin(w * x) / w)
        train = design[:n]
        coef = np.linalg.solve(train.T @ train + reg, train.T @ targets)
        return np.mean((train @ coef - targets)**2), design, coef

    grid = np.geomspace(0.01, 2.6, 29)
    idx = int(np.argmin([fit(k)[0] for k in grid]))
    lo, hi = grid[max(idx - 1, 0)], grid[min(idx + 1, len(grid) - 1)]
    phi = (np.sqrt(5.) - 1.) / 2.
    a, b = hi - phi * (hi - lo), lo + phi * (hi - lo)
    fa, fb = fit(a), fit(b)
    for _ in range(24):
        if fa[0] < fb[0]:
            hi, b, fb = b, a, fa
            a = hi - phi * (hi - lo)
            fa = fit(a)
        else:
            lo, a, fa = a, b, fb
            b = lo + phi * (hi - lo)
            fb = fit(b)
    _, design, coef = min(fa, fb, key=lambda out: out[0])
    return design[n:] @ coef


def _predict_scalar(x, y):
    """Profile saturation and prune weak terms using training data only."""
    n = len(y)
    best = None
    for k in np.geomspace(0.02, 2.0, 60):
        w = np.sqrt(k)
        design = _cubic(np.tanh(w * x) / w)
        train = design[:n]
        coef = np.linalg.lstsq(train, y, rcond=None)[0]
        keep = np.ones(design.shape[1], dtype=bool)
        for _ in range(6):
            new_keep = keep & (np.abs(coef) > 0.035)
            new_keep[:x.shape[1] + 1] = True
            if np.array_equal(new_keep, keep):
                break
            keep = new_keep
            coef = np.zeros(design.shape[1])
            coef[keep] = np.linalg.lstsq(train[:, keep], y, rcond=None)[0]
        score = np.mean((train @ coef - y)**2) + 1e-5 * keep.sum()
        if best is None or score < best[0]:
            best = (score, design, coef)
    _, design, coef = best
    return design[n:] @ coef


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    upstream = inputs['node_00']
    train = study['train']
    # The frozen hybrid appends a cubic basis in panel-A-calibrated
    # coordinates. Its linear block gives those coordinates directly.
    dim = train['x'].shape[1]
    offset = upstream['aligned'].shape[1]
    x = upstream['hybrid'][:, offset + 1:offset + 1 + dim]
    y = _predict_scalar(x, train['y'])
    z = train['traj']
    temporal = np.linalg.svd(z, full_matrices=False)[2][:2]
    state = _predict(x, z @ temporal.T, 0.0001) @ temporal
    diagnostics = np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
