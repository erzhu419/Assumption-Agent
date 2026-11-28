"""Estimate the calibration, threshold operator, and two observation laws."""
import numpy as np
from scipy.optimize import least_squares


def _fit(x, y):
    return np.linalg.lstsq(x, y, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = [np.asarray(panel[k], dtype=float) for k in ('u', 'v')]
    keep = np.ones(len(u), dtype=bool)
    for _ in range(8):
        if keep.all():
            # Initialize using a robust, row-wise discrepancy from identity.
            dev = np.linalg.norm(v - u - np.median(v - u, axis=0), axis=1)
            keep = dev <= max(5 * np.median(dev), 1e-7)
        uc = u[keep] - u[keep].mean(axis=0)
        vc = v[keep] - v[keep].mean(axis=0)
        gain = np.sum(uc * vc, axis=0) / np.maximum(np.sum(uc * uc, axis=0), 1e-15)
        offset = np.mean(v[keep] - u[keep] * gain, axis=0)
        residual = np.linalg.norm(v - (offset + u * gain), axis=1)
        refined = residual <= max(5 * np.median(residual), 1e-7)
        if refined.sum() < 8 or np.array_equal(keep, refined):
            break
        keep = refined
    return offset, gain


def _warp(x, p):
    return np.sign(x) * np.maximum(np.abs(x) - p[0], 0.0) ** p[1]


def _operator(panel):
    u, v = [np.asarray(panel[k], dtype=float) for k in ('u', 'v')]

    def residual(p):
        a = _affine(_warp(u, p))
        return (a @ _fit(a, v) - v).ravel()

    scale = np.max(np.abs(u))
    p = least_squares(residual, [0.25 * scale, 1.5],
                      bounds=([0.0, 0.5], [0.8 * scale, 3.5]),
                      ftol=1e-11, xtol=1e-11, gtol=1e-11).x
    return p, _fit(_affine(_warp(u, p)), v)


def _readout(w):
    return np.column_stack((w, w[:, 0] * w[:, 2] + w[:, 1] * w[:, 3]))


def _relaxation(w, wt, observed, times):
    # Profile out the spatial coefficients before fitting the two decay rates.
    q = np.linalg.qr(w, mode='reduced')[0]
    target = (q.T @ observed).T

    def basis(p):
        return -np.expm1(-np.outer(times, p))

    def residual(p):
        b = basis(p)
        return (b @ _fit(b, target) - target).ravel()

    rate = least_squares(residual, [0.4, 1.8],
                         bounds=([0.005, 0.6], [0.8, 25.0]),
                         ftol=1e-10, xtol=1e-10, gtol=1e-10).x
    b = basis(rate)
    amplitudes = _fit(b, observed.T).T
    return wt @ _fit(w, amplitudes) @ b.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    offset, gain = _calibration(study['panel_a'])
    params, mix = _operator(study['panel_b'])

    def encode(x):
        return _affine(_warp((np.asarray(x, dtype=float) - offset) / gain, params)) @ mix

    w, wt = encode(train['x']), encode(study['test']['x'])
    y_hat = _readout(wt) @ _fit(_readout(w), np.asarray(train['y'], dtype=float))
    state = _relaxation(w, wt, np.asarray(train['traj'], dtype=float),
                        np.asarray(study['eval_times'], dtype=float))
    last = state[:, -1]
    return {'y_hat': y_hat, 'state_pred': state,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
