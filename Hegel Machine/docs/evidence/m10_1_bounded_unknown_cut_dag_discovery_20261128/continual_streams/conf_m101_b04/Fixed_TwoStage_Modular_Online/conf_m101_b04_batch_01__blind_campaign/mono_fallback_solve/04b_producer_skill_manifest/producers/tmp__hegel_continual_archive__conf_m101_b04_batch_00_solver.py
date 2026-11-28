"""Fit separate nonlinear scalar and trajectory readouts from the study."""
from itertools import combinations_with_replacement

import numpy as np
from scipy.linalg import cho_factor, cho_solve, qr
from scipy.optimize import least_squares, minimize


def _monomials(x, degree):
    return np.column_stack([np.prod(x[:, cols], axis=1) for cols in
                            combinations_with_replacement(range(x.shape[1]), degree)])


def _quadratic(x):
    return np.column_stack((np.ones(len(x)), x, _monomials(x, 2)))


def _gaussian(x, z):
    return np.exp(-0.23 * np.sum((x[:, None] - z[None, :]) ** 2, axis=2))


def _predict(p, pt, k, kt, target, penalty):
    f = cho_factor(k + penalty * np.eye(len(k)))
    a = cho_solve(f, p)
    v = cho_solve(f, target)
    trend = np.linalg.solve(p.T @ a, p.T @ v)
    return pt @ trend + kt @ (v - a @ trend)


def _matern_readout(x, xt, p, pt, y):
    # Profile out the quadratic mean before fitting the input length scales.
    n = qr(p, mode='full')[0][:, p.shape[1]:]
    v = n.T @ y
    delta = (x[:, None] - x[None, :]) ** 2
    delta_test = (xt[:, None] - x[None, :]) ** 2

    def kernel(distance_sq, par):
        r2 = distance_sq @ np.exp(par[:x.shape[1]])
        r = np.sqrt(r2)
        return (1 + r + 0.4 * r2 + r2 * r / 15) * np.exp(-r)

    def objective(par):
        k = n.T @ kernel(delta, par) @ n + np.exp(par[-1]) * np.eye(len(v))
        f = cho_factor(k)
        return np.log(v @ cho_solve(f, v) / len(v)) + 2 * np.log(np.diag(f[0])).mean()

    par = minimize(objective, np.log([0.24] * x.shape[1] + [0.002]),
                   method='L-BFGS-B', bounds=[(-4, 1)] * x.shape[1] + [(-12, -1)],
                   options={'maxiter': 70}).x
    return _predict(p, pt, kernel(delta, par), kernel(delta_test, par), y, np.exp(par[-1]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    x, xt = train['x'], study['test']['x']
    t = study['eval_times']
    trajectories = train['traj'].T

    def basis(rates):
        return -np.expm1(-t[:, None] * rates)

    def residual(rates):
        b = basis(rates)
        amplitudes = np.linalg.lstsq(b, trajectories, rcond=None)[0]
        return (b @ amplitudes - trajectories).ravel()

    rates = least_squares(residual, [1.5, 0.25],
                          bounds=([0.7, 0.02], [4.0, 0.65])).x
    modes = np.linalg.qr(basis(rates))[0]
    p, pt = _quadratic(x), _quadratic(xt)
    k, kt = _gaussian(x, x), _gaussian(xt, x)
    y = train['y']
    y_hat = 0.5 * (_predict(p, pt, k, kt, y, 0.001)
                   + _matern_readout(x, xt, p, pt, y))
    c, ct = _monomials(x, 3), _monomials(xt, 3)
    amplitudes = _predict(p, pt, k + 0.05 * (c @ c.T),
                          kt + 0.05 * (ct @ c.T), trajectories.T @ modes, 0.006)
    state_pred = amplitudes @ modes.T
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
