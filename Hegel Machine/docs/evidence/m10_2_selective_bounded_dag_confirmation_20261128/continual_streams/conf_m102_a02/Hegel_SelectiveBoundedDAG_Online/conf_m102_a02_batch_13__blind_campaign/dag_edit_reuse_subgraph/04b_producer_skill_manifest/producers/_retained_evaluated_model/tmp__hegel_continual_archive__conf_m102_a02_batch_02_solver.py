"""Calibrated tilted-gate scalar regression and low-rank affine state regression."""
from itertools import combinations_with_replacement
import numpy as np
from scipy.optimize import least_squares


def _calibration(panel):
    u, v = panel['u'], panel['v']
    delta = np.linalg.norm(v - u, axis=1)
    med = np.median(delta)
    keep = delta <= med + max(0.25, 8 * np.median(abs(delta - med)))
    u, v = u[keep], v[keep]
    uc, vc = u - u.mean(0), v - v.mean(0)
    slope = (uc * vc).sum(0) / (uc * uc).sum(0)
    return v.mean(0) - slope * u.mean(0), slope


def _affine(x):
    return np.column_stack((np.ones(len(x)), x))


def _quadratic(x):
    return np.column_stack((_affine(x), *(x[:, i] * x[:, j] for i, j in combinations_with_replacement(range(4), 2))))


def _gated(x, q, params):
    slope, center = params[:2]
    score = slope * (x[:, 2] - center) + x[:, [0, 1, 3]] @ params[2:]
    return np.column_stack((q, q[:, :5] * np.tanh(score)[:, None]))


def _predict_scalar(x, y, test):
    q = _quadratic(x)

    # Profile the linear readout out of the nonlinear gate fit.
    def residual(params):
        design = _gated(x, q, params)
        return design @ _regress(design, y) - y

    # Select the gate by training residual only. Multiple starts avoid a
    # steep, displaced gate that can look locally optimal on smaller folds.
    starts = ([8, 0, 0, 0, 0], [3, 0, 0, 0, 0], [14, 0, 0, 0, 0],
              [8, -.12, 0, 0, 0], [8, .12, 0, 0, 0],
              [8, 0, 1, 0, 0], [8, 0, -1, 0, 0])
    fit = min((least_squares(
        residual, start,
        bounds=([1, -.35, -3, -3, -3], [35, .35, 3, 3, 3]),
        max_nfev=100,
    ) for start in starts), key=lambda result: result.cost)
    qt = _quadratic(test)
    coef = _regress(_gated(x, q, fit.x), y)
    tilted = _gated(test, qt, fit.x) @ coef

    # A small fixed-gate component reduces split-to-split prediction variance.
    fixed = np.column_stack((q, q * np.tanh(8 * x[:, 2:3])))
    fixed_test = np.column_stack((qt, qt * np.tanh(8 * test[:, 2:3])))
    return .8 * tilted + .2 * (fixed_test @ _regress(fixed, y))


def _regress(a, b):
    return np.linalg.lstsq(a, b, rcond=None)[0]


def solve_batch(study: dict) -> dict:
    train = study['train']
    x = np.asarray(train['x'], dtype=float)
    test = np.asarray(study['test']['x'], dtype=float)
    offset, slope = _calibration(study['panel_a'])
    y_hat = _predict_scalar((x - offset) / slope, train['y'], (test - offset) / slope)
    a = _affine(x)
    time_basis = np.linalg.svd(a @ _regress(a, train['traj']), full_matrices=False)[2][:2].T
    state_pred = (_affine(test) @ _regress(a, train['traj'] @ time_basis)) @ time_basis.T
    return {'y_hat': y_hat, 'state_pred': state_pred, 'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), state_pred[:, -1].mean(), state_pred[:, -1].std()])}
