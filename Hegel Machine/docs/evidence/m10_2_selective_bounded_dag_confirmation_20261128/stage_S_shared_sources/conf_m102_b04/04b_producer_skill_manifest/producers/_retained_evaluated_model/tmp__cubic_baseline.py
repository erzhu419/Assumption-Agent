import numpy as np
from itertools import combinations_with_replacement


def _basis(x):
    return np.column_stack([np.ones(len(x))] + [np.prod(x[:, j], axis=1) for degree in (1, 2, 3) for j in combinations_with_replacement(range(x.shape[1]), degree)])


def solve_batch(study: dict) -> dict:
    tr = study['train']
    X = _basis(tr['x'])
    Q = _basis(study['test']['x'])
    z = tr['traj']
    basis = np.linalg.svd(z, full_matrices=False)[2][:2]
    design = X.T @ X + .006 * np.eye(X.shape[1])
    fit = np.linalg.solve(design, X.T @ np.column_stack((tr['y'], z @ basis.T)))
    pred = Q @ fit
    y, state = pred[:, 0], pred[:, 1:] @ basis
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': np.array([y.mean(), y.std(), state[:, -1].mean(), state[:, -1].std()])}
