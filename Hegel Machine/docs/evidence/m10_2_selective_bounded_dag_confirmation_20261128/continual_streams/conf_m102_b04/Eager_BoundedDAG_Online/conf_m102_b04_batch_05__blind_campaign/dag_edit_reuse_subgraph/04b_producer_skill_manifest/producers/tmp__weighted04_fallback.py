"""Fit scalar and low-rank trajectory heads on the frozen calibration."""
from itertools import combinations_with_replacement
import numpy as np


def _design(q, scale):
    # Saturation controls extrapolation; penalize high-degree mixed terms more.
    x = q / np.sqrt(1.0 + scale * q * q)
    cols = [np.ones(len(x))]
    for degree in range(1, 7):
        for j in combinations_with_replacement(range(x.shape[1]), degree):
            weight = 0.5 ** (len(set(j)) - 1 + 0.5 * max(degree - 2, 0))
            cols.append(weight * np.prod(x[:, j], axis=1))
    return np.column_stack(cols)


def _predict(f, target, reg):
    n = len(target)
    train = f[:n]
    gram = train @ train.T
    gram.flat[::n + 1] += reg
    return f[n:] @ train.T @ np.linalg.solve(gram, target)


def run_node(inputs: dict) -> dict:
    study = inputs['__input__']
    features = inputs['node_00']
    q = features['hybrid'][:, features['aligned'].shape[1] + 1:features['aligned'].shape[1] + 5]
    targets = study['train']
    traj = targets['traj']
    modes = np.linalg.svd(traj, full_matrices=False)[2][:2]
    y = _predict(_design(q, 0.60), targets['y'], 0.0003)
    state = _predict(_design(q, 0.85), traj @ modes.T, 0.00146) @ modes
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([
        y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
