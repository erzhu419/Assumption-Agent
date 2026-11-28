"""Calibrate the cyclic sensor array and apply the reference-coordinate map."""
import numpy as np
from scipy.optimize import least_squares


def _fit_sensor(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    nxt = np.roll(u, -1, axis=1)
    d = np.stack((np.ones_like(u), u, nxt, u*nxt, u**3), axis=2)
    keep = np.max(np.abs(v-u), axis=1) < 3.0
    return np.stack([np.linalg.lstsq(d[keep, j], v[keep, j], rcond=None)[0]
                     for j in range(u.shape[1])], axis=1)


def _invert(x, coeff):
    offset, gain, coupling, product, cubic = coeff
    n = x.shape[1]
    i = np.arange(n)
    z = (x-offset)/gain
    for _ in range(18):
        nxt = np.roll(z, -1, axis=1)
        r = offset + gain*z + coupling*nxt + product*z*nxt + cubic*z**3 - x
        if np.max(np.abs(r), initial=0.0) < 1e-11:
            break
        jac = np.zeros((len(x), n, n))
        jac[:, i, i] = gain + product*nxt + 3*cubic*z*z
        jac[:, i, (i+1)%n] = coupling + product*z
        step = np.linalg.solve(jac, r[..., None])[..., 0]
        z -= np.clip(step, -0.5, 0.5)
    return z


def _design(x, log_scale):
    return np.column_stack((np.ones(len(x)), x, x / (1 + np.exp(log_scale)*x*x)))


def _fit_coordinates(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def residual(log_scale):
        a = _design(u, log_scale)
        return (a @ np.linalg.lstsq(a, v, rcond=None)[0] - v).ravel()
    log_scale = least_squares(residual, np.zeros(u.shape[1]), bounds=(-3., 3.), max_nfev=100).x
    weights = np.linalg.lstsq(_design(u, log_scale), v, rcond=None)[0]
    return log_scale, weights


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    coeff = _fit_sensor(study['panel_a'])
    scale, weights = _fit_coordinates(study['panel_b'])
    def coordinates(part):
        z = _invert(np.asarray(study[part]['x'], dtype=float), coeff)
        return _design(z, scale) @ weights
    return {'c': coordinates('train'), 'query': coordinates('test')}
