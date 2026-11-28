"""Polynomial baseline for the public scientific prediction task."""
import numpy as np


def _design(x):
    x = np.asarray(x, dtype=float)
    i, j = np.triu_indices(x.shape[1])
    return np.column_stack((np.ones(len(x)), x, x[:, i] * x[:, j]))


def solve_batch(study: dict) -> dict:
    train = study['train']
    target = np.column_stack((train['y'], train['traj']))
    coeff = np.linalg.lstsq(_design(train['x']), target, rcond=None)[0]
    pred = _design(study['test']['x']) @ coeff
    y, state = pred[:, 0], pred[:, 1:]
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
