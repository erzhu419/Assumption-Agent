"""Predict scalar and trajectory outputs from the calibrated operator coordinates."""

import numpy as np
from scipy.optimize import least_squares


def _ls(design, response):
    return np.linalg.lstsq(design, response, rcond=None)[0]


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _coordinates(study, x):
    a = study['panel_a']
    u, v = a['u'], a['v']
    designs = [_affine(u[:, j:j+1]) for j in range(u.shape[1])]
    keep = np.arange(len(u))
    for step in range(7):
        coeffs = np.array([_ls(d[keep], v[keep, j]) for j, d in enumerate(designs)])
        residuals = np.linalg.norm(v - np.column_stack(
            [d @ c for d, c in zip(designs, coeffs)]), axis=1)
        if step < 6:
            keep = np.argsort(residuals)[:int(0.85 * len(u))]
    keep = residuals <= 5 * np.median(residuals)
    coeffs = np.array([_ls(d[keep], v[keep, j]) for j, d in enumerate(designs)])
    b = study['panel_b']
    operator = _ls(_affine(b['u']), b['v'])
    return _affine((x - coeffs[:, 0]) / coeffs[:, 1]) @ operator


def _trajectory(train_x, train_z, test_x, times):
    coeffs = _ls(train_x, train_z)
    fitted = train_x @ coeffs

    def project(rates):
        basis = -np.expm1(-times[:, None] * rates)
        return _ls(basis, coeffs.T).T @ basis.T

    rates = least_squares(
        lambda p: (train_x @ project(p) - fitted).ravel(),
        [0.42, 1.82], bounds=([0.02, 0.9], [0.9, 4.0]),
    ).x
    return test_x @ project(rates)


def solve_batch(study: dict) -> dict:
    train, test = study['train'], study['test']
    u = _coordinates(study, train['x'])
    v = _coordinates(study, test['x'])

    def scalar_design(w):
        interaction = w[:, 0] * w[:, 2] + w[:, 1] * w[:, 3]
        return np.column_stack((w, interaction))

    y_hat = scalar_design(v) @ _ls(scalar_design(u), train['y'])
    state_pred = _trajectory(u, train['traj'], v, study['eval_times'])
    end = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])}
