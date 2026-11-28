"""Predict scalar and trajectory readouts in inverse-calibrated coordinates."""

import numpy as np
from scipy.optimize import least_squares, minimize_scalar


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _fit(design, target):
    return np.linalg.lstsq(design, target, rcond=None)[0]


def _reference(panel, measured):
    u, v = panel['u'], panel['v']
    affine = _design(u)
    residual = np.linalg.norm(affine @ _fit(affine, v) - v, axis=1)
    keep = np.argsort(residual)[:45]

    def columns(x):
        return np.stack((np.ones_like(x), x, np.roll(x, -1, axis=1) * (1 + x), x**3), axis=-1)

    features = columns(u)
    for _ in range(4):
        weights = np.stack([_fit(features[keep, j], v[keep, j]) for j in range(u.shape[1])])
        residual = np.linalg.norm(np.einsum('nij,ij->ni', features, weights) - v, axis=1)
        keep = np.argsort(residual)[:45]

    def forward(x):
        return np.sum(columns(x[None])[0] * weights, axis=1)

    return np.array([least_squares(lambda x: forward(x) - row, row).x for row in measured])


def _panel_map(u, v, x, design, bounds):
    def residual(rate):
        matrix = design(u, rate)
        return np.sum((matrix @ _fit(matrix, v) - v)**2)

    rate = minimize_scalar(residual, bounds=bounds, method='bounded').x
    return design(x, rate) @ _fit(design(u, rate), v)


def _odd(x, rate):
    return _design(-np.sign(x) * np.expm1(-rate * np.abs(x)))


def _even(x, rate):
    return _design(x, x*x / np.sqrt(1 + rate*x*x))


def _ridge_predict(features, target, penalty):
    n = len(target)
    train = features[:n]
    regularizer = np.eye(train.shape[1]) * penalty
    regularizer[0, 0] = 0
    return features[n:] @ np.linalg.solve(train.T @ train + regularizer, train.T @ target)


def solve_batch(study: dict) -> dict:
    train = study['train']
    measured = np.concatenate((train['x'], study['test']['x']))
    a = _reference(study['panel_a'], measured)
    panel = study['panel_b']
    b = _design(a) @ _fit(_design(panel['u']), panel['v'])
    panel = study['panel_c']
    c = _panel_map(panel['u'], panel['v1'], b, _odd, (0.1, 5.0))
    d = _panel_map(panel['u'], panel['v2'], b, _even, (0.0001, 2.0))
    scalar_design = _design(a, c, d, *(np.sum(left * right, axis=1)
                                       for left, right in ((a, c), (b, c), (a, d), (b, d), (c, d))))
    y = _ridge_predict(scalar_design, train['y'], 0.001)

    cross = (a[:, :, None] * c[:, None, :]).reshape(len(a), -1)
    state_design = _design(a, c, d, cross)
    t = np.asarray(study['eval_times'])
    trajectories = train['traj'].T

    def modes(rates):
        return -np.expm1(-t[:, None] * rates)

    def temporal_residual(rates):
        basis = modes(rates)
        return (basis @ _fit(basis, trajectories) - trajectories).ravel()

    rates = least_squares(temporal_residual, [0.4, 1.8], bounds=([0.05, 0.95], [0.95, 5.0])).x
    basis = modes(rates)
    state = _ridge_predict(state_design, _fit(basis, trajectories).T, 0.001) @ basis.T
    terminal = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), terminal.mean(), terminal.std()])}
