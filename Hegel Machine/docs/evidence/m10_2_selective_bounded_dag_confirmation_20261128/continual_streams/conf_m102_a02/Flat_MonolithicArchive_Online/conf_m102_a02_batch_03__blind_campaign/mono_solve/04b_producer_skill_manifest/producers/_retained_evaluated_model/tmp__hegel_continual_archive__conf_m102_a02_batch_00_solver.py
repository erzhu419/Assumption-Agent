"""Quadratic scalar readout and rank-two affine trajectory model."""
from itertools import combinations_with_replacement

import numpy as np


def _design(x):
    x = np.asarray(x, dtype=float)
    affine = np.column_stack((np.ones(len(x)), x))
    quadratic = [x[:, i] * x[:, j]
                 for i, j in combinations_with_replacement(range(x.shape[1]), 2)]
    return affine, np.column_stack((affine, *quadratic))


def solve_batch(study: dict) -> dict:
    train = study['train']
    a, q = _design(train['x'])
    a_test, q_test = _design(study['test']['x'])

    # The endpoint needs all pairwise products, including squares. A full
    # quadratic span is invariant under invertible affine panel calibration.
    readout = np.linalg.lstsq(q, train['y'], rcond=None)[0]
    y = q_test @ readout

    # Estimate the temporal span from fitted signals, not raw noisy snapshots.
    state_coef = np.linalg.lstsq(a, train['traj'], rcond=None)[0]
    _, _, modes = np.linalg.svd(a @ state_coef, full_matrices=False)
    basis = modes[:2].T
    state = (a_test @ state_coef @ basis) @ basis.T
    final = state[:, -1]
    diagnostics = np.array([y.mean(), y.std(), final.mean(), final.std()])
    return {'y_hat': y, 'state_pred': state, 'diagnostics_vec': diagnostics}
