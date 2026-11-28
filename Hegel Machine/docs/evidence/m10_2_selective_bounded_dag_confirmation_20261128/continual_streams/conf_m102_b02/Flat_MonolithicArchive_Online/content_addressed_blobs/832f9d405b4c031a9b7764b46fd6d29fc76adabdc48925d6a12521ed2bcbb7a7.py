"""Fit panel operators and separate polynomial response and trajectory readouts."""
from itertools import combinations_with_replacement

import numpy as np


def _with_intercept(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=1e-10)[0]


def _calibration(panel):
    u, v = panel['u'], panel['v']
    d = _with_intercept(u)
    keep = np.ones(len(u), dtype=bool)
    for _ in range(10):
        coef = _ls(d[keep], v[keep])
        residual = np.linalg.norm(d @ coef - v, axis=1)
        take = residual <= max(.001, 3 * np.median(residual))
        if take.sum() < d.shape[1] or np.array_equal(keep, take):
            break
        keep = take
    coef = _ls(d[keep], v[keep])
    scale = np.linalg.norm(coef[1:])
    if np.linalg.norm(coef[1:] - np.diag(np.diag(coef[1:]))) < .001 * scale:
        for j in range(u.shape[1]):
            coef[[0, j + 1], j] = _ls(d[keep][:, [0, j + 1]], v[keep, j])
        coef[1:] = np.diag(np.diag(coef[1:]))
    return coef


def _features(study, x):
    a = _calibration(study['panel_a'])
    z = np.linalg.solve(a[1:].T, (x - a[0]).T).T
    b = study['panel_b']
    w = _with_intercept(z) @ _ls(_with_intercept(b['u']), b['v'])
    c = study['panel_c']
    i, j = np.array(list(combinations_with_replacement(range(w.shape[1]), 2))).T
    p = c['u'][:, i] * c['u'][:, j]
    coef = _ls(_with_intercept(p), c['v1'])
    norms = np.linalg.norm(coef[1:], axis=1)
    active = norms > .001 * norms.max()
    if not np.any(active):
        active[:] = True
    coef = _ls(_with_intercept(p[:, active]), c['v1'])
    q = (w[:, i[active]] * w[:, j[active]]) @ coef[1:]
    base = _with_intercept(np.column_stack((w, q)))
    # The response couples the two coordinates through a shared dot product.
    response = np.column_stack((base, np.sum(z * q, axis=1)))
    interaction = (z[:, :, None] * q[:, None, :]).reshape(len(z), -1)
    state = np.column_stack((base, interaction))
    return response, state


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    dy, ds = _features(study, np.vstack((train['x'], study['test']['x'])))
    y = dy[n:] @ _ls(dy[:n], train['y'])
    traj = np.asarray(train['traj'])
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    state = (ds[n:] @ _ls(ds[:n], traj @ modes.T)) @ modes
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
