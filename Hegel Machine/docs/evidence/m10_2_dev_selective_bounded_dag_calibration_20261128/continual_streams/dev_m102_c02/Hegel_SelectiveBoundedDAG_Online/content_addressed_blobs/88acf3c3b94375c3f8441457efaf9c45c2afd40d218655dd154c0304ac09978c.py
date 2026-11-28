"""Calibrate raw measurements using the two paired reference panels."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _panel_basis(x, curvature):
    rational = x / (1.0 + curvature * x * x)
    return np.column_stack((_affine(x), rational))


def _panel_calibration(u, v):
    """Profile out linear mixing while estimating positive channel curvatures."""
    u, v = np.asarray(u, dtype=float), np.asarray(v, dtype=float)

    def residual(curvature):
        design = _panel_basis(u, curvature)
        weights = np.linalg.lstsq(design, v, rcond=None)[0]
        return (design @ weights - v).ravel()

    curvature = least_squares(residual, np.full(u.shape[1], 2.0),
                              bounds=(0.01, 30.0), max_nfev=70).x
    weights = np.linalg.lstsq(_panel_basis(u, curvature), v, rcond=None)[0]
    return curvature, weights


def _channel_calibration(u, v):
    offset, gain = np.empty(u.shape[1]), np.empty(u.shape[1])
    i, k = np.triu_indices(len(u), 1)
    for j, (a, b) in enumerate(zip(u.T, v.T)):
        delta = a[k] - a[i]
        ok = np.abs(delta) > .1*np.ptp(a)
        slope = np.median((b[k[ok]] - b[i[ok]])/delta[ok])
        bias = np.median(b - a*slope)
        err = np.abs(b - (bias + a*slope))
        keep = err <= max(6*np.median(err), 1e-10)
        if keep.sum() >= 2 and np.ptp(a[keep]) > 0:
            bias, slope = np.linalg.lstsq(_affine(a[keep, None]), b[keep], rcond=None)[0]
        offset[j], gain[j] = bias, slope
    return offset, gain


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    a, b = study['panel_a'], study['panel_b']
    offset, gain = _channel_calibration(a['u'], a['v'])
    curvature, operator = _panel_calibration(b['u'], b['v'])

    def coordinates(x):
        return _panel_basis((np.asarray(x, dtype=float) - offset)/gain, curvature) @ operator

    return {'c': coordinates(study['train']['x']), 'query': coordinates(study['test']['x'])}
