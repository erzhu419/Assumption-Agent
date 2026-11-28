"""Fit a shared cubic regression to the calibrated coordinates and target modes."""
import numpy as np


def run_node(inputs: dict) -> dict:
    coordinates = inputs['node_00']
    projection = inputs['node_01']
    z = np.asarray(coordinates['z'], dtype=float)
    zt = np.asarray(coordinates['zt'], dtype=float)
    targets = np.asarray(projection['targets'], dtype=float)
    modes = np.asarray(projection['modes'], dtype=float)

    # The homogeneous kernel expansion includes all monomials through degree 3.
    scale = np.maximum(z.std(axis=0), 1e-10)
    x, xt = z / scale, zt / scale
    d = x.shape[1]
    gram = (1.0 + (x @ x.T) / d) ** 3
    gram.flat[::len(x) + 1] += 1e-4
    pred = (1.0 + (xt @ x.T) / d) ** 3 @ np.linalg.solve(gram, targets)

    y = pred[:, 0].copy()
    state = pred[:, 1:] @ modes
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
