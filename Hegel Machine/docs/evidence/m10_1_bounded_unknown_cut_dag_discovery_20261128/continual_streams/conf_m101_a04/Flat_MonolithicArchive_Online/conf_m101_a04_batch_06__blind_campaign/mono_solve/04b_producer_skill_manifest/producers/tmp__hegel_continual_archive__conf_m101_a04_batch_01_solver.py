"""Calibrated panel-C operators with separate scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        a, b = u[keep], v[keep]
        am, bm = a.mean(axis=0), b.mean(axis=0)
        ac, bc = a - am, b - bm
        scale = (ac * bc).sum(axis=0) / (ac ** 2).sum(axis=0)
        offset = bm - scale * am
        error = np.linalg.norm(u * scale + offset - v, axis=1)
        updated = error < max(5 * np.median(error), 0.0006)
        if np.array_equal(keep, updated):
            break
        keep = updated
    return offset, scale


def _bilinear_basis(x):
    a, b, c, d = x.T
    return np.column_stack((np.ones(len(x)), a*b, a*d, b*c, c*d))


def _cosine_basis(x, frequency):
    return np.column_stack((np.ones(len(x)), x, np.cos(x * frequency)))


def _branches(panel, x):
    u = panel['u']
    c = _bilinear_basis(x) @ _fit(_bilinear_basis(u), panel['v1'])

    def residual(frequency):
        design = _cosine_basis(u, frequency)
        return (design @ _fit(design, panel['v2']) - panel['v2']).ravel()

    frequency = least_squares(residual, np.full(u.shape[1], 1.29), bounds=(0.6, 2.0)).x
    d = _cosine_basis(x, frequency) @ _fit(_cosine_basis(u, frequency), panel['v2'])
    return c, d


def _trajectory(g, h, values, g_test, h_test):
    # Each branch group has its own amplitude and shared temporal profile.
    width = g.shape[1]
    initial = _fit(np.column_stack((g, h)), values)
    f = np.linalg.svd(initial[:width], full_matrices=False)[2][0]
    j = np.linalg.svd(initial[width:], full_matrices=False)[2][0]
    for _ in range(50):
        gf = (g[:, None, :] * f[None, :, None]).reshape(-1, width)
        hj = (h[:, None, :] * j[None, :, None]).reshape(-1, h.shape[1])
        weights = _fit(np.column_stack((gf, hj)), values.ravel())
        amplitudes = np.column_stack((g @ weights[:width], h @ weights[width:]))
        f, j = _fit(amplitudes, values)
    return np.outer(g_test @ weights[:width], f) + np.outer(h_test @ weights[width:], j)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    offset, scale = _calibrate(study['panel_a'])
    raw = np.concatenate((train['x'], study['test']['x']), axis=0)
    c, d = _branches(study['panel_c'], (raw - offset) / scale)
    matched = c * d
    scalar = np.column_stack((c, d, matched.sum(axis=1)))
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])
    g = np.column_stack((c, matched))
    h = np.column_stack((d, c[:, 0] * d[:, 1]))
    state = _trajectory(g[:n], h[:n], train['traj'], g[n:], h[n:])
    last = state[:, -1]
    return {
        'y_hat': y_hat,
        'state_pred': state,
        'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()]),
    }
