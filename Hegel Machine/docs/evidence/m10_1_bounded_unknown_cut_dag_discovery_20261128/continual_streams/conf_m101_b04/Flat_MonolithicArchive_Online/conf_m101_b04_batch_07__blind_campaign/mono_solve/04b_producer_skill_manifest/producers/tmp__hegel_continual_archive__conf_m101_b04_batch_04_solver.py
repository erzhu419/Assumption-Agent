"""Invert the reference calibration and fit the scalar and relaxation readouts."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares, root


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _quadratic(z):
    return np.column_stack((_affine(z), *(z[:, j] * z[:, k] for j, k in
                                            combinations_with_replacement(range(z.shape[1]), 2))))


def _calibrate(panel, observed):
    u, v = panel['u'], panel['v']
    next_u = np.roll(u, -1, axis=1)
    designs = [np.column_stack((np.ones(len(u)), a, b, a * b, a**3))
               for a, b in zip(u.T, next_u.T)]
    # The sparse corruptions are much larger than the calibration nonlinearity.
    affine = _affine(u)
    keep = np.arange(len(u))
    for _ in range(3):
        residual = np.linalg.norm(affine @ _fit(affine[keep], v[keep]) - v, axis=1)
        keep = np.flatnonzero(residual < 5 * np.median(residual))
    coef = np.array([_fit(p[keep], v[keep, j]) for j, p in enumerate(designs)]).T

    def forward(z):
        w = np.roll(z, -1)
        return coef[0] + coef[1]*z + coef[2]*w + coef[3]*z*w + coef[4]*z**3

    return np.array([root(lambda z: forward(z) - row, row, tol=1e-10).x
                     for row in observed])


def _time_basis(t, traj):
    def basis(rates):
        return -np.expm1(-t[:, None] * rates)

    def residual(rates):
        b = basis(rates)
        return (b @ _fit(b, traj.T) - traj.T).ravel()

    result = least_squares(residual, [1.8, .41], bounds=([.7, .05], [4., .65]))
    return np.linalg.qr(basis(result.x))[0]


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    n = len(train['x'])
    observed = np.concatenate((train['x'], test['x']))
    design = _quadratic(_calibrate(study['panel_a'], observed))
    basis = _time_basis(study['eval_times'], train['traj'])
    target = np.column_stack((train['y'], train['traj'] @ basis))
    prediction = design[n:] @ _fit(design[:n], target)
    y_hat = prediction[:, 0]
    state_pred = prediction[:, 1:] @ basis.T
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
