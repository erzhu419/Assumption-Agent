"""Calibrate the panel operators and fit a two-mode trajectory model."""

import numpy as np
from scipy.optimize import least_squares


def _design(*blocks):
    return np.column_stack((np.ones(len(blocks[0])), *blocks))


def _fit(design, values):
    return np.linalg.lstsq(design, values, rcond=None)[0]


def _reference(panel):
    u, v = panel['u'], panel['v']
    offset = np.empty(u.shape[1])
    scale = np.empty_like(offset)
    for j in range(u.shape[1]):
        design = _design(u[:, j])
        values = v[:, j]
        beta = _fit(design, values)
        for _ in range(8):
            keep = np.argsort(np.abs(design @ beta - values))[:int(0.8 * len(u))]
            beta = _fit(design[keep], values[keep])
        offset[j], scale[j] = beta
    return offset, scale


def _shrink(x):
    return np.sign(x) * np.maximum(np.abs(x) - 0.35, 0.0)**1.5


def _pair_basis(x):
    return _design(x[:, [0, 0, 1, 2]] * x[:, [1, 3, 2, 3]])


def _mode_basis(features, times, rates):
    modes = -np.expm1(-times[:, None] * rates)
    return (modes[None, :, :, None] * features[:, None, None, :]).reshape(
        len(features) * len(times), 2 * features.shape[1]
    )


def solve_batch(study: dict) -> dict:
    """Predict the scalar response, sampled trajectory, and output moments."""
    train = study['train']
    x = np.vstack((train['x'], study['test']['x']))
    n = len(train['x'])
    times = np.asarray(study['eval_times'])

    offset, scale = _reference(study['panel_a'])
    a = (x - offset) / scale

    panel = study['panel_b']
    u, v = _shrink(panel['u']), panel['v']
    uc, vc = u - u.mean(axis=0), v - v.mean(axis=0)
    left, singular, right = np.linalg.svd(uc.T @ vc)
    transform = (left @ right) * (singular.sum() / np.sum(uc**2))
    b = (_shrink(a) - u.mean(axis=0)) @ transform + v.mean(axis=0)

    panel = study['panel_c']
    c = _pair_basis(b) @ _fit(_pair_basis(panel['u']), panel['v1'])
    shared = _design(a, c, np.sum(a * c, axis=1))
    y_hat = shared[n:] @ _fit(shared[:n], train['y'])

    features = np.column_stack((shared, a[:, 1] * c[:, 0]))
    observed = np.asarray(train['traj']).ravel()

    def residual(rates):
        design = _mode_basis(features[:n], times, rates)
        return design @ _fit(design, observed) - observed

    # Bounds preserve fast/slow ordering while profiling out readout weights.
    rates = least_squares(residual, [1.8, 0.4], bounds=([1.0, 0.1], [3.0, 0.9])).x
    weights = _fit(_mode_basis(features[:n], times, rates), observed)
    state_pred = (_mode_basis(features[n:], times, rates) @ weights).reshape(len(x) - n, len(times))
    last = state_pred[:, -1]
    return {
        'y_hat': y_hat,
        'state_pred': state_pred,
        'diagnostics_vec': np.array([y_hat.mean(), y_hat.std(), last.mean(), last.std()]),
    }
