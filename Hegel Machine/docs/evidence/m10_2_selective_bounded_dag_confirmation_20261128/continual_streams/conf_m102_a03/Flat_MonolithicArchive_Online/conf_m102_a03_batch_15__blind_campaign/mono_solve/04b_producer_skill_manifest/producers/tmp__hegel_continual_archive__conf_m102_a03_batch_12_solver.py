"""Panel-calibrated cosine operator with scalar and trajectory readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibration(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    mask = np.ones(len(u), dtype=bool)
    for _ in range(6):
        bias, gain = np.array([_fit(_affine(a[mask]), b[mask]) for a, b in zip(u.T, v.T)]).T
        residual = np.linalg.norm(v - (bias + u * gain), axis=1)
        next_mask = residual <= max(4 * np.median(residual), 1e-8)
        if np.array_equal(mask, next_mask) or next_mask.sum() < 8:
            break
        mask = next_mask
    return bias, gain


def _operator(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    def design(z, frequency):
        return _affine(np.cos(frequency * z))
    def loss(frequency):
        a = design(u, frequency)
        return np.sum((a @ _fit(a, v) - v)**2)
    grid = np.linspace(.2, 16., 160)
    i = np.argmin([loss(k) for k in grid])
    frequency = minimize_scalar(loss, bounds=(grid[max(0, i-1)], grid[min(len(grid)-1, i+1)]), method='bounded', options={'xatol': 1e-11}).x
    mix = _fit(design(u, frequency), v)
    return frequency, mix


def _scalar_design(w):
    pair_sum = w[:, 0] * w[:, 2] + w[:, 1] * w[:, 3]
    return _affine(np.column_stack((w, pair_sum)))


def _trajectory(w, wt, trajectories, times):
    x = _affine(w)
    projected = (np.linalg.qr(x, mode='reduced')[0].T @ trajectories).T

    def basis(rates):
        return -np.expm1(-np.outer(times, rates))

    def residual(rates):
        a = basis(rates)
        return (a @ _fit(a, projected) - projected).ravel()

    rates = least_squares(residual, [.4, 1.8], bounds=([.005, .6], [.8, 15.])).x
    a = basis(rates)
    amplitudes = _fit(a, trajectories.T).T
    return (_affine(wt) @ _fit(x, amplitudes)) @ a.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    bias, gain = _calibration(study['panel_a'])
    frequency, mix = _operator(study['panel_b'])
    def encode(x):
        return _affine(np.cos(frequency * ((np.asarray(x) - bias) / gain))) @ mix
    w, wt = encode(train['x']), encode(study['test']['x'])
    y_hat = _scalar_design(wt) @ _fit(_scalar_design(w), train['y'])
    state_pred = _trajectory(w, wt, np.asarray(train['traj']), np.asarray(study['eval_times']))
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
