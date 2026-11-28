"""Predict scalar and temporal outcomes from the two calibrated panel-C branches."""
import numpy as np
from scipy.optimize import minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        a, b = u[keep], v[keep]
        ac, bc = a - a.mean(0), b - b.mean(0)
        scale = (ac * bc).sum(0) / (ac * ac).sum(0)
        offset = b.mean(0) - a.mean(0) * scale
        residual = np.linalg.norm(u * scale + offset - v, axis=1)
        updated = residual < max(5 * np.median(residual), .0006)
        if np.array_equal(updated, keep):
            break
        keep = updated
    return offset, scale


def _panel_basis(x, w, branch):
    h = x / (1 + w * np.abs(x)) if branch == 'v1' else np.cos(w * x)
    return np.column_stack((np.ones(len(x)), x, h))


def _operator(panel, x, branch):
    u, v = panel['u'], panel[branch]

    def residual(w):
        a = _panel_basis(u, w, branch)
        return np.mean((a @ _fit(a, v) - v)**2)

    bounds = (.3, 1.6) if branch == 'v1' else (1.8, 3.)
    w = minimize_scalar(residual, method='bounded', bounds=bounds).x
    return _panel_basis(x, w, branch) @ _fit(_panel_basis(u, w, branch), v)


def _design(study):
    offset, scale = _calibrate(study['panel_a'])
    raw = np.concatenate((study['train']['x'], study['test']['x']))
    x = (raw - offset) / scale
    c = _operator(study['panel_c'], x, 'v1')
    d = _operator(study['panel_c'], x, 'v2')
    matched = (c * d).sum(axis=1)
    scalar = np.column_stack((c, d, matched))
    first = np.column_stack((c, matched))
    second = np.column_stack((d, c[:, 0] * d[:, 1]))
    return scalar, first, second


def _trajectory(g, h, values, g_test, h_test):
    # Each branch uses one shared temporal profile, with its own sample amplitudes.
    width = g.shape[1]
    initial = _fit(np.column_stack((g, h)), values)
    f = np.linalg.svd(initial[:width], full_matrices=False)[2][0]
    j = np.linalg.svd(initial[width:], full_matrices=False)[2][0]
    for _ in range(40):
        gf = (g[:, None, :] * f[None, :, None]).reshape(-1, width)
        hj = (h[:, None, :] * j[None, :, None]).reshape(-1, h.shape[1])
        weights = _fit(np.column_stack((gf, hj)), values.ravel())
        amplitudes = np.column_stack((g @ weights[:width], h @ weights[width:]))
        f, j = _fit(amplitudes, values)
    return np.outer(g_test @ weights[:width], f) + np.outer(h_test @ weights[width:], j)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    scalar, g, h = _design(study)
    y_hat = scalar[n:] @ _fit(scalar[:n], train['y'])
    state = _trajectory(g[:n], h[:n], train['traj'], g[n:], h[n:])
    last = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
