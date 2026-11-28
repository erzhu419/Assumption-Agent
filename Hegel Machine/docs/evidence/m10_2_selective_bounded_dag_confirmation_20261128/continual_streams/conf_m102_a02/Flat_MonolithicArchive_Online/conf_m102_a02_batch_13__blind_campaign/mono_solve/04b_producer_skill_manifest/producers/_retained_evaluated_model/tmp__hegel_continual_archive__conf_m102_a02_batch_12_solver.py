"""Infer the calibrated cosine operator and its scalar and temporal readouts."""
import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _fit(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def _aff(x):
    return np.column_stack((np.ones(len(x)), x))


def _calibrate(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    i, j = np.triu_indices(len(u), 1)
    du = u[i] - u[j]
    slopes = np.divide(v[i] - v[j], du, out=np.full_like(du, np.nan), where=np.abs(du) > 1e-10)
    scale = np.nanmedian(slopes, axis=0)
    shift = np.median(v - u * scale, axis=0)
    for j in range(u.shape[1]):
        residual = np.abs(v[:, j] - u[:, j] * scale[j] - shift[j])
        keep = residual <= max(0.001, 6 * np.median(residual))
        shift[j], scale[j] = _fit(_aff(u[keep, j]), v[keep, j])
    return shift, scale


def _operator(panel):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))

    def design(x, frequency):
        return _aff(np.cos(frequency * x))

    def loss(frequency):
        d = design(u, frequency)
        return np.sum((d @ _fit(d, v) - v) ** 2)

    grid = np.linspace(0.2, 8.0, 80)
    i = int(np.argmin([loss(k) for k in grid]))
    frequency = minimize_scalar(loss, bounds=(grid[max(0, i - 1)], grid[min(len(grid)-1, i + 1)]),
                                method='bounded', options={'xatol': 1e-10}).x
    w = _fit(design(u, frequency), v)
    return lambda x: design(x, frequency) @ w


def _quadratic(x):
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((_aff(x), x[:, i] * x[:, j]))


def _trajectory(h, q, t, traj):
    # Regress spatial amplitudes and constrain their time courses to two
    # shared exponential relaxations, with zero displacement at time zero.
    a = _aff(h)
    projected = a @ _fit(a, traj)

    def basis(rates):
        return -np.expm1(-t[:, None] * rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _fit(b, projected.T) - projected.T).ravel()

    rates = least_squares(residual, [0.4, 1.8], bounds=([0.02, 0.02], [10., 10.])).x
    b = basis(rates)
    amplitudes = _fit(b, traj.T).T
    return _aff(q) @ _fit(a, amplitudes) @ b.T


def solve_batch(study: dict) -> dict:
    train = study['train']
    shift, scale = _calibrate(study['panel_a'])
    op = _operator(study['panel_b'])
    h = op((np.asarray(train['x']) - shift) / scale)
    q = op((np.asarray(study['test']['x']) - shift) / scale)
    y = _quadratic(q) @ _fit(_quadratic(h), np.asarray(train['y']))
    state = _trajectory(h, q, np.asarray(study['eval_times']), np.asarray(train['traj']))
    last = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), last.mean(), last.std()]) if len(y) else np.zeros(4)
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
