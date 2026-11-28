"""Predict scalar and trajectory readouts in calibrated panel coordinates."""
from itertools import combinations_with_replacement

import numpy as np


def _affine_design(u):
    return np.column_stack((np.ones(len(u)), u))


def _calibrate(panel, measured):
    u, v = (np.asarray(panel[k], dtype=float) for k in ('u', 'v'))
    corrected = np.empty_like(measured, dtype=float)
    for j in range(u.shape[1]):
        d = _affine_design(u[:, j:j+1])
        keep = np.ones(len(u), bool)
        for _ in range(8):
            coef = np.linalg.lstsq(d[keep], v[keep, j], rcond=None)[0]
            residual = np.abs(d @ coef - v[:, j])
            new = residual <= max(.0005, 4 * np.median(residual))
            if new.sum() < 3 or np.array_equal(new, keep):
                break
            keep = new
        corrected[:, j] = (measured[:, j] - coef[0]) / coef[1]
    return corrected


def _design(z):
    pairs = ((0, 1), (0, 3), (1, 2), (2, 3))
    triples = [c for c in combinations_with_replacement(range(4), 3)
               if any(i in c and j in c for i, j in pairs)]
    return np.column_stack((np.ones(len(z)), z, *(
        np.prod(z[:, c], axis=1) for c in (*pairs, *triples))))


def solve_batch(study: dict) -> dict:
    train = study['train']
    n = len(train['x'])
    x = _calibrate(study['panel_a'], np.vstack((train['x'], study['test']['x'])))
    b = study['panel_b']
    operator = np.linalg.lstsq(_affine_design(b['u']), b['v'], rcond=None)[0]
    d = _design(_affine_design(x) @ operator)
    target = np.column_stack((train['y'], train['traj']))
    coef = np.linalg.lstsq(d[:n], target, rcond=None)[0]
    pred = d[n:] @ coef
    y, state = pred[:, 0], pred[:, 1:]
    # Estimate the shared two-mode time subspace using only training trajectories.
    modes = np.linalg.svd(train['traj'], full_matrices=False)[2][:2]
    state = (state @ modes.T) @ modes
    end = state[:, -1]
    return dict(y_hat=y, state_pred=state,
                diagnostics_vec=np.array([y.mean(), y.std(), end.mean(), end.std()]))
