import numpy as np
from scipy.optimize import root


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _calibrate(a, frequency):
    u, v = a['u'], a['v']
    design = np.column_stack((_aff(u), np.sin(frequency * u)))
    magnitude = np.max(np.abs(v), axis=1)
    keep = magnitude <= 2.5 * np.median(magnitude)
    if keep.sum() < design.shape[1] + 2:
        keep = np.ones(len(u), dtype=bool)
    for _ in range(5):
        q = _ls(design[keep], v[keep])
        residual = np.linalg.norm(design @ q - v, axis=1)
        new_keep = residual <= max(0.015, 5 * np.median(residual))
        if new_keep.sum() < design.shape[1] + 2 or np.array_equal(keep, new_keep):
            break
        keep = new_keep
    return q


def _invert(x, q, frequency):
    dim = x.shape[1]
    linear, sine = q[1:1+dim], q[1+dim:]
    initial = np.linalg.solve(linear.T, (x - q[0]).T).T

    def jac(t):
        return (linear + (frequency * np.cos(frequency * t))[:, None] * sine).T

    def invert_row(observed, guess):
        def fun(t):
            return q[0] + t @ linear + np.sin(frequency * t) @ sine - observed
        fit = root(fun, guess, jac=jac, tol=1e-10)
        if np.linalg.norm(fit.fun, ord=np.inf) > 1e-8:
            from scipy.optimize import least_squares
            fit = least_squares(fun, fit.x, jac=jac, gtol=1e-12)
        return fit.x

    return np.stack([invert_row(obs, guess) for obs, guess in zip(x, initial)]) if len(x) else initial


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    frequency = float(study['telemetry_vec'][0])
    x = np.concatenate((study['train']['x'], study['test']['x']), axis=0)
    u = _invert(x, _calibrate(study['panel_a'], frequency), frequency)
    b = study['panel_b']
    return {'w': _aff(u) @ _ls(_aff(b['u']), b['v'])}
