"""Calibrate the recorded coordinates and fit separate label and trajectory readouts."""

import numpy as np
from scipy.optimize import least_squares


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _calibrate(panel):
    u, v = panel['u'], panel['v']
    keep = np.linalg.norm(u - v, axis=1) < 2
    return np.array([_fit(_affine(a[keep]), b[keep]) for a, b in zip(u.T, v.T)]).T


def _operator(panel):
    u, v = panel['u'], panel['v']

    def design(x, q):
        k, a = np.exp(q)
        return _affine(x * (1 - a * np.exp(-k * x**2)))

    def residual(q):
        d = design(u, q)
        return (d @ _fit(d, v) - v).ravel()

    q = least_squares(residual, np.log([0.7, 1.06]), bounds=(-4, 2)).x
    weights = _fit(design(u, q), v)
    return lambda x: design(x, q) @ weights


def _label_design(z):
    a, b, c, d = z.T
    return np.column_stack((_affine(z), a * c, b * d))


def _time_basis(times, trajectories):
    def basis(log_rates):
        return -np.expm1(-np.exp(log_rates)[:, None] * times)

    def residual(log_rates):
        q = basis(log_rates)
        return (trajectories @ np.linalg.pinv(q) @ q - trajectories).ravel()

    solution = least_squares(residual, np.log([0.4, 1.8]), bounds=(-4, 3))
    return basis(solution.x)


def solve_batch(study: dict) -> dict:
    train = study['train']
    offset, gain = _calibrate(study['panel_a'])
    operator = _operator(study['panel_b'])
    z = operator((train['x'] - offset) / gain)
    test_z = operator((study['test']['x'] - offset) / gain)

    y_hat = _label_design(test_z) @ _fit(_label_design(z), train['y'])
    q = _time_basis(study['eval_times'], train['traj'])
    amplitudes = train['traj'] @ np.linalg.pinv(q)
    state_pred = _affine(test_z) @ _fit(_affine(z), amplitudes) @ q
    last = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()])}
