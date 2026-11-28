"""Fit a nonlinear scalar readout and a shared two-mode trajectory."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares


def _quadratic(x):
    return np.column_stack([np.ones(len(x)), x] +
                           [x[:, i] * x[:, j] for i, j in
                            combinations_with_replacement(range(x.shape[1]), 2)])


def _kernel(x, z):
    return np.exp(-0.23 * np.sum((x[:, None] - z[None, :]) ** 2, axis=2))


def _readout(x, xt, target, penalty):
    p, pt = _quadratic(x), _quadratic(xt)
    k = _kernel(x, x) + penalty * np.eye(len(x))
    system = np.block([[k, p], [p.T, np.zeros((p.shape[1], p.shape[1]))]])
    rhs = np.concatenate((target, np.zeros((p.shape[1],) + target.shape[1:])))
    coef = np.linalg.solve(system, rhs)
    return np.column_stack((_kernel(xt, x), pt)) @ coef


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = train['x'], study['test']['x']
    t = study['eval_times']
    trajectories = train['traj'].T

    def basis(rates):
        return -np.expm1(-t[:, None] * rates)

    def project(b):
        return np.linalg.lstsq(b, trajectories, rcond=None)[0]

    def residual(rates):
        b = basis(rates)
        return (b @ project(b) - trajectories).ravel()

    rates = least_squares(residual, [1.5, 0.25],
                          bounds=([0.7, 0.02], [4.0, 0.65])).x
    b = basis(rates)
    y_hat = _readout(x, xt, train['y'], 0.001)
    state_pred = _readout(x, xt, project(b).T, 0.006) @ b.T
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
