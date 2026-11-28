"""Independent panel calibration and low-rank dynamical readout."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _ls(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _calibrate(panel, x):
    coeff = []
    for u, v in zip(panel['u'].T, panel['v'].T):
        d = _aff(u)
        rows = np.arange(len(u))
        keep = max(2, int(0.84 * len(u)))
        for _ in range(12):
            err = np.abs(d @ _ls(d[rows], v[rows]) - v)
            rows = np.argsort(err)[:keep]
        inlier = err < max(0.001, 8 * np.median(err))
        coeff.append(_ls(d[inlier], v[inlier]))
    offset, scale = np.asarray(coeff).T
    return (x - offset) / scale


def _operator(panel, key, feature, initial, bounds):
    u, v = panel['u'], panel[key]
    def residual(p):
        d = _aff(feature(u, p))
        return (d @ _ls(d, v) - v).ravel()
    p = least_squares(residual, initial, bounds=bounds).x
    weights = _ls(_aff(feature(u, p)), v)
    return lambda x: _aff(feature(x, p)) @ weights


def _coordinates(study, x):
    z = _calibrate(study['panel_a'], x)
    b = _operator(study['panel_b'], 'v', lambda u, p: np.cos(p[0] * u), [2.6], ([1.5], [3.5]))
    def c_features(u, p):
        return np.column_stack((u / (1 + p[0] * u*u), np.arcsinh(p[1] * u)))
    c = _operator(study['panel_c'], 'v1', c_features, [.22, 2.15], ([.06, .06], [6., 6.]))
    return c(b(z))


def _scalar_design(q):
    pairs = [q[:, i] * q[:, j] for i, j in combinations_with_replacement(range(q.shape[1]), 2)]
    return _aff(np.column_stack((q, *pairs)))


def _trajectory(d, dt, times, observations):
    def basis(rates):
        return -np.expm1(-times[:, None] * rates)
    # Profile the spatial loadings first; estimate rates only from predictable variation.
    coeff = _ls(d, observations)
    fitted = d @ coeff
    def residual(rates):
        b = basis(rates)
        return (b @ _ls(b, fitted.T) - fitted.T).ravel()
    rates = least_squares(residual, [.4, 1.8], bounds=([.05, 1.], [.99, 4.])).x
    b = basis(rates)
    return dt @ (_ls(b, coeff.T).T @ b.T)


def solve_batch(study: dict) -> dict:
    tr = study['train']
    n = len(tr['x'])
    q = _coordinates(study, np.concatenate((tr['x'], study['test']['x'])))
    d = _scalar_design(q)
    y = d[n:] @ _ls(d[:n], tr['y'])
    d = _aff(q)
    state = _trajectory(d[:n], d[n:], np.asarray(study['eval_times']), tr['traj'])
    tail = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), tail.mean(), tail.std()])}
