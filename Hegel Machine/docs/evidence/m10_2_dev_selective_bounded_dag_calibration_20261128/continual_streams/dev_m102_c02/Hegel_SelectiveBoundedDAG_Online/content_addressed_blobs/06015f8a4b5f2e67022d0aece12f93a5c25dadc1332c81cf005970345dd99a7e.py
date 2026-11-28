"""Align observed coordinates to panel A, then apply the panel-B cosine map."""
import numpy as np
from scipy.optimize import minimize_scalar


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    offset, gain = np.empty(u.shape[1]), np.empty(u.shape[1])
    for k in range(u.shape[1]):
        x, y = u[:, k], v[:, k]
        dx = x[j] - x[i]
        keep = np.abs(dx) > 0.1 * np.ptp(x)
        slope = np.median((y[j[keep]] - y[i[keep]]) / dx[keep])
        bias = np.median(y - slope * x)
        error = np.abs(y - slope * x - bias)
        inlier = error <= max(6 * np.median(error), 1e-9)
        offset[k], gain[k] = np.linalg.lstsq(_affine(x[inlier, None]), y[inlier], rcond=None)[0]
    return offset, gain


def _operator(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(x, frequency):
        return _affine(np.cos(frequency * x))

    def fit(frequency):
        d = design(u, frequency)
        beta = np.linalg.lstsq(d, v, rcond=None)[0]
        return beta, np.mean((v - d @ beta) ** 2)

    grid = np.linspace(0.1, 12.0, 120)
    i = np.argmin([fit(w)[1] for w in grid])
    frequency = minimize_scalar(lambda w: fit(w)[1], method='bounded',
                                bounds=(grid[max(0, i - 1)], grid[min(i + 1, len(grid) - 1)]),
                                options={'xatol': 1e-10}).x
    beta, _ = fit(frequency)
    return frequency, beta


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    offset, gain = _calibrate(study['panel_a'])
    frequency, beta = _operator(study['panel_b'])

    def coordinates(x):
        z = (np.asarray(x, dtype=float) - offset) / gain
        return _affine(np.cos(frequency * z)) @ beta

    return {'c': coordinates(study['train']['x']),
            'query': coordinates(study['test']['x'])}
