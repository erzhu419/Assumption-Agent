"""Predict the scalar readout and time course from panel-calibrated inputs."""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _odd_basis(x, parameters):
    h = np.log1p(np.abs(x) * np.exp(parameters[:4]))
    return _design(x, np.sign(x) * h * (1 + parameters[4] * h))


def _even_basis(x, rate):
    return _design(x, np.sqrt(1 + rate * x**2) - 1)


def _features(study, x):
    b = study['panel_b']
    f = _design(x) @ _fit(_design(b['u']), b['v'])
    panel = study['panel_c']
    u, v, w = (panel[k] for k in ('u', 'v1', 'v2'))

    def odd_residual(parameters):
        g = _odd_basis(u, parameters)
        return (g @ _fit(g, v) - v).ravel()

    p = least_squares(odd_residual, [.4] * 4 + [.1], bounds=(-3, 3),
                      ftol=1e-12, gtol=1e-12).x
    c = _odd_basis(f, p) @ _fit(_odd_basis(u, p), v)

    def even_error(rate):
        g = _even_basis(u, rate)
        return np.linalg.norm(g @ _fit(g, w) - w)

    q = minimize_scalar(even_error, bounds=(.0001, 1.8), method='bounded').x
    d = _even_basis(f, q) @ _fit(_even_basis(u, q), w)
    mixed = (f[:, :, None] * c[:, None, :]).reshape(len(f), 16)
    return _design(f, c, d, mixed)


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    design = _features(study, np.row_stack((train['x'], study['test']['x'])))
    x = design[:n]
    traj = train['traj']
    basis = np.linalg.svd(traj, full_matrices=False)[2][:2]
    target = np.column_stack((train['y'], traj @ basis.T))
    penalty = np.full(x.shape[1], .0003)
    penalty[0] = 0
    beta = np.linalg.solve(x.T @ x + np.diag(penalty), x.T @ target)
    prediction = design[n:] @ beta
    y_hat = prediction[:, 0]
    state_pred = prediction[:, 1:] @ basis
    terminal = state_pred[:, -1]
    return {'y_hat': y_hat, 'state_pred': state_pred,
            'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(),
                                         terminal.mean(), terminal.std()])}
