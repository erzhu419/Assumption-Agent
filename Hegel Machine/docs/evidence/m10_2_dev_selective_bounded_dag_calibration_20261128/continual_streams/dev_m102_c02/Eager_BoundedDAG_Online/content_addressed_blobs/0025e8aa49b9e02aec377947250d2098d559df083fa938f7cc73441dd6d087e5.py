"""Recover instrument calibration and the shared cosine coordinate map."""
import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _instrument(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    offset, gain = np.empty(u.shape[1]), np.empty(u.shape[1])
    i, j = np.triu_indices(len(u), 1)
    for k, (a, b) in enumerate(zip(u.T, v.T)):
        dx = a[j] - a[i]
        keep = np.abs(dx) > 0.1 * np.ptp(a)
        slope = np.median((b[j[keep]] - b[i[keep]]) / dx[keep])
        bias = np.median(b - slope * a)
        r = b - bias - slope * a
        mad = np.median(np.abs(r))
        good = np.abs(r) <= max(6.0 * mad, 1e-8 * max(1.0, np.ptp(b)))
        offset[k], gain[k] = np.linalg.lstsq(_affine(a[good, None]), b[good], rcond=None)[0]
    return offset, gain


def _cosine_map(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(w):
        return _affine(np.cos(w * u))

    def objective(w):
        h = design(w)
        r = h @ np.linalg.lstsq(h, v, rcond=None)[0] - v
        return np.sum(r * r)

    grid = np.arange(0.25, 12.01, 0.125)
    w = grid[np.argmin([objective(w) for w in grid])]
    w = minimize_scalar(objective, bounds=(w - 0.125, w + 0.125),
                        method='bounded', options={'xatol': 1e-9}).x
    return w, np.linalg.lstsq(design(w), v, rcond=None)[0]


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    offset, gain = _instrument(study['panel_a'])
    w, coef = _cosine_map(study['panel_b'])

    def transform(split):
        x = np.asarray(study[split]['x'], dtype=float)
        return _affine(np.cos(w * ((x - offset) / gain))) @ coef

    return {'c': transform('train'), 'query': transform('test')}
