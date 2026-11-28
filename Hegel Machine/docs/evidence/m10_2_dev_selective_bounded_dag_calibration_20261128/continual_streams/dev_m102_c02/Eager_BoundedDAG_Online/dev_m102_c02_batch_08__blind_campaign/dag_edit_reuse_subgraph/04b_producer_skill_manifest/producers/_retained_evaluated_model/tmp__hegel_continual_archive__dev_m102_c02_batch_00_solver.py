"""Recover calibrated coordinates, a bilinear readout, and two relaxation modes."""
import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _panel_maps(panel_a, panel_b):
    u, v = (np.asarray(panel_a[k], dtype=float) for k in ('u', 'v'))
    offset, gain = np.empty(u.shape[1]), np.empty(u.shape[1])
    for j in range(u.shape[1]):
        a, b = u[:, j], v[:, j]
        i, k = np.triu_indices(len(a), 1)
        dx = a[k] - a[i]
        ok = np.abs(dx) > 0.1 * np.ptp(a)
        slope = np.median((b[k[ok]] - b[i[ok]]) / dx[ok])
        bias = np.median(b - a * slope)
        residual = b - (bias + slope * a)
        mad = np.median(np.abs(residual))
        keep = np.abs(residual) <= max(6 * mad, 1e-10)
        if keep.sum() >= 2 and np.ptp(a[keep]) > 0:
            bias, slope = np.linalg.lstsq(_affine(a[keep, None]), b[keep], rcond=None)[0]
        offset[j], gain[j] = bias, slope
    u, v = (np.asarray(panel_b[k], dtype=float) for k in ('u', 'v'))
    operator = np.linalg.lstsq(_affine(u), v, rcond=None)[0]
    return offset, gain, operator


def _coordinates(x, maps):
    offset, gain, operator = maps
    return _affine((x - offset) / gain) @ operator


def _readout(c, intercept=True):
    return np.column_stack((_affine(c) if intercept else c,
                            np.sum(c[:, :2] * c[:, 2:], axis=1)))


def _trajectory(c, query, traj, times, intercept=True):
    design, new_design = (_affine(c), _affine(query)) if intercept else (c, query)
    beta = np.linalg.lstsq(design, traj, rcond=None)[0]
    fitted = design @ beta
    scale = max(float(np.max(np.abs(times))), np.finfo(float).tiny)
    t = times / scale

    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates[:, None]) * t)

    def residual(log_rates):
        modes = basis(log_rates)
        return (fitted @ np.linalg.pinv(modes) @ modes - fitted).ravel()

    log_rates = least_squares(residual, np.log([0.6, 2.7]),
                              bounds=(np.log([0.03, 0.25]), np.log([4., 20.])),
                              max_nfev=100).x
    modes = basis(log_rates)
    return new_design @ (beta @ np.linalg.pinv(modes)) @ modes


def solve_batch(study: dict) -> dict:
    train = study['train']
    maps = _panel_maps(study['panel_a'], study['panel_b'])
    c = _coordinates(np.asarray(train['x'], dtype=float), maps)
    query = _coordinates(np.asarray(study['test']['x'], dtype=float), maps)
    beta = np.linalg.lstsq(_readout(c), np.asarray(train['y'], dtype=float), rcond=None)[0]
    y_hat = _readout(query) @ beta
    state_pred = _trajectory(c, query, np.asarray(train['traj'], dtype=float),
                             np.asarray(study['eval_times'], dtype=float))
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
