"""Fit separate latent, folded, and mixed terms to the shared response."""
import numpy as np
from scipy.linalg import solve


def _normalized(train, test):
    center = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-10)
    return (train - center) / scale, (test - center) / scale


def _kernel(z, b, other_z, other_b):
    a = 0.15 * (z @ other_z.T) / z.shape[1]
    c = (b @ other_b.T) / b.shape[1]
    return (1 + a)**2 + (1 + c)**2 + a * (c + 0.5*c*c)


def run_node(inputs: dict) -> dict:
    coords, projection = inputs['node_00'], inputs['node_01']
    z, zt = _normalized(coords['z_train'], coords['z_test'])
    b, bt = _normalized(coords['b_train'], coords['b_test'])
    targets, modes = projection['targets'], projection['modes']
    kernel = _kernel(z, b, z, b)
    kernel.flat[::len(z)+1] += 0.001
    estimated = _kernel(zt, bt, z, b) @ solve(kernel, targets, assume_a='pos')
    y = estimated[:, 0].copy()
    state = estimated[:, 1:] @ modes
    end = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), end.mean(), end.std()])}
