from itertools import combinations_with_replacement

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _affine(z):
    return np.column_stack((np.ones(len(z)), z))


def _quadratic(z):
    return np.column_stack((_affine(z), *[z[:, i] * z[:, j] for i, j in combinations_with_replacement(range(z.shape[1]), 2)]))


def _readout(u, v, z):
    def design(a, w):
        return np.column_stack((_affine(a), np.cos(w * a)))

    def fit(w):
        a = design(u, w)
        return a, np.linalg.lstsq(a, v, rcond=None)[0]

    def loss(w):
        a, coef = fit(w)
        return np.mean((a @ coef - v) ** 2)

    w = minimize_scalar(loss, bounds=(0.6, 2.2), method='bounded').x
    _, coef = fit(w)
    return design(z, w) @ coef


def _kernel(a, b):
    d = ((a[:, None] - b[None, :]) ** 2).sum(axis=2)
    return np.exp(-0.5 * d)


def _response_trend(train_basis, test_basis, y):
    # Profile out the affine trend and the amplitude of the saturating term.
    affine = train_basis[:, :5]
    terms = train_basis[:, 1:]
    orth = np.linalg.qr(affine)[0]
    target = y - orth @ (orth.T @ y)

    def loss(p):
        h = np.tanh(p[0] + terms @ p[1:])
        h -= orth @ (orth.T @ h)
        amplitude = (h @ target) / (h @ h + 1e-10)
        return np.r_[amplitude * h - target, 0.002 * p]

    direction = np.linalg.lstsq(train_basis, y, rcond=None)[0]
    rng = np.random.default_rng(15)
    starts = [direction, *rng.normal(0, 0.8, (7, len(direction)))]
    p = min((least_squares(loss, g, max_nfev=90) for g in starts),
            key=lambda fit: fit.cost).x

    def design(basis):
        return np.column_stack((basis[:, :5], np.tanh(p[0] + basis[:, 1:] @ p[1:])))

    a = design(train_basis)
    coef = np.linalg.lstsq(a, y, rcond=None)[0]
    return y - a @ coef, design(test_basis) @ coef


def run_node(inputs: dict) -> dict:
    study, calibrated = inputs['__input__'], inputs['node_00']
    q, qt = calibrated['q_tr'], calibrated['q_te']
    train, panel = study['train'], study['panel_c']
    basis, test_basis = _quadratic(q), _quadratic(qt)
    residual, trend = _response_trend(basis, test_basis, train['y'])
    z = np.vstack((q, qt))
    z = np.column_stack((z, 0.2 * _readout(panel['u'], panel['v2'], z)))
    n = len(q)
    k = _kernel(z, z[:n])
    y_hat = trend + k[n:] @ np.linalg.solve(k[:n] + 0.003 * np.eye(n), residual)
    state_pred = test_basis @ np.linalg.lstsq(basis, train['traj'], rcond=None)[0]
    end = state_pred[:, -1]
    diagnostics_vec = np.array([y_hat.mean(), y_hat.std(), end.mean(), end.std()])
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': diagnostics_vec}
