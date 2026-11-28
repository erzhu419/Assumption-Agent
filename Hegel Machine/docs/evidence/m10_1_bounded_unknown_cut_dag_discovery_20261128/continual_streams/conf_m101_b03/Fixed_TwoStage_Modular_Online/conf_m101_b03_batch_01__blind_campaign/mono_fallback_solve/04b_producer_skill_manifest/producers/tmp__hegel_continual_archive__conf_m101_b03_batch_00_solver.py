"""Predict the response and two-mode relaxation trajectory."""
from itertools import combinations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import least_squares, minimize


_PAIRS = tuple(combinations(range(4), 2))


def _kernel(a, b, h, mode):
    d = np.moveaxis((a[:, None] - b[None, :]) ** 2 * np.exp(2 * h[:4]), -1, 0)
    dot = 1 + a @ b.T
    cov = np.exp(h[4]) * dot + np.exp(h[5]) * dot**2

    def matern5(s):
        r = np.sqrt(5 * s)
        return (1 + r + r*r/3) * np.exp(-r)

    if mode == 3:
        r = np.sqrt(3 * d.sum(axis=0))
        cov += np.exp(h[6]) * (1 + r) * np.exp(-r)
        start = 7
    else:
        cov += np.exp(h[6]) * matern5(d.sum(axis=0))
        start = 11
        if mode == 1:
            for w, di in zip(h[7:11], d):
                cov += np.exp(w) * matern5(di)
    if mode in (1, 3):
        for w, (i, j) in zip(h[start:], _PAIRS):
            cov += np.exp(w) * matern5(d[i] + d[j])
    return cov


def _regress(x, y, test, mode):
    n = len(x)
    mean, std = y.mean(), y.std()
    if std == 0:
        return np.full(len(test), mean)
    z = (y - mean) / std
    nh = {0: 7, 1: 17, 3: 13}[mode]

    def fit(h):
        k = _kernel(x, x, h, mode)
        k.flat[::n+1] += np.exp(h[-1])
        c = cho_factor(k)
        return c, cho_solve(c, z)

    def loss(h):
        c, w = fit(h)
        return .5 * z @ w + np.log(np.diag(c[0])).sum()

    h = np.r_[np.full(4, -.7), -1., -3., -.5, np.full(nh-7, -3.), -8.]
    opt = minimize(loss, h, method='L-BFGS-B',
                   bounds=[(-5, 3)]*4 + [(-12, 5)]*(nh-4) + [(-14, -1)],
                   options={'maxiter': 120})
    _, w = fit(opt.x)
    return mean + std * (_kernel(test, x, opt.x, mode) @ w)


def _time_modes(t, traj):
    def basis(r):
        return -np.expm1(-t[:, None] * r)

    def project(h):
        return np.linalg.lstsq(h, traj.T, rcond=None)[0]

    def residual(r):
        h = basis(r)
        return (h @ project(h) - traj.T).ravel()

    rates = least_squares(residual, [.5, 2.], bounds=(.01, 10.)).x
    h = basis(np.sort(rates))
    return h, project(h).T


def solve_batch(study):
    tr = study['train']
    x, test = tr['x'], study['test']['x']
    h, modes = _time_modes(study['eval_times'], tr['traj'])
    y = _regress(x, tr['y'], test, 3)
    a = _regress(x, modes[:, 0], test, 0)
    b = _regress(x, modes[:, 1], test, 1)
    state = np.c_[a, b] @ h.T
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
