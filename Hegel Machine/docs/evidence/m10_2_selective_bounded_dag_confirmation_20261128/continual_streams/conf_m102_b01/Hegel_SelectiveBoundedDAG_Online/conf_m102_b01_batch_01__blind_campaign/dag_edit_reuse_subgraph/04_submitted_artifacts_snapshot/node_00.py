"""Estimate the shared latent coordinates from independent calibration panels."""
import numpy as np
from scipy.optimize import minimize_scalar


def _lstsq(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _sensor_calibration(u, v):
    offsets = np.empty(u.shape[1])
    scales = np.empty_like(offsets)
    for j, x in enumerate(u.T):
        y = v[:, j]
        design = np.column_stack((np.ones(len(x)), x))
        beta = _lstsq(design, y)
        retain = max(2, int(0.875 * len(x)))
        for _ in range(8):
            selected = np.argsort(np.abs(design @ beta - y))[:retain]
            beta = _lstsq(design[selected], y[selected])
        residual = np.abs(design @ beta - y)
        clean = residual <= max(0.002, 10 * np.median(residual))
        if np.count_nonzero(clean) >= 2:
            beta = _lstsq(design[clean], y[clean])
        offsets[j], scales[j] = beta
    return offsets, scales


def _basis(x, curvature):
    x2 = x * x
    return np.column_stack((np.ones(len(x)), x, x * x2 / (1.0 + curvature * x2)))


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    panel_a, panel_b = study['panel_a'], study['panel_b']
    offsets, scales = _sensor_calibration(panel_a['u'], panel_a['v'])
    u, v = panel_b['u'], panel_b['v']

    def loss(curvature):
        design = _basis(u, curvature)
        return np.mean((design @ _lstsq(design, v) - v) ** 2)

    grid = np.linspace(0.0, 5.0, 31)
    i = int(np.argmin([loss(c) for c in grid]))
    fit = minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(i+1, len(grid)-1)]), method='bounded')
    curvature = min((grid[i], fit.x), key=loss)
    beta = _lstsq(_basis(u, curvature), v)
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    return {'w': _basis((x - offsets) / scales, curvature) @ beta}
