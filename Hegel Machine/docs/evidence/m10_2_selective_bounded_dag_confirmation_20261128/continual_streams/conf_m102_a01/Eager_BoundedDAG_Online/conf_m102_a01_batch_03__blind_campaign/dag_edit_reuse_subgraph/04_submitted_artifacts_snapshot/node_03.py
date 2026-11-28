"""Combine frozen branch predictions and reconstruct the trajectory."""
import numpy as np


def run_node(inputs: dict) -> dict:
    poly = np.asarray(inputs['node_01']['poly'])
    kernel = inputs['node_02']
    smooth = np.asarray(kernel['smooth'])
    vt = np.asarray(kernel['vt'])

    # The kernel anchors the scalar estimate; the polynomial can temper
    # its downward shrinkage without importing large downward excursions.
    delta = poly[:, 0] - smooth[:, 0]
    y = smooth[:, 0] + 0.20 * np.maximum(delta, 0.0)

    state = (0.4 * poly[:, 1:] + 0.6 * smooth[:, 1:]) @ vt
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
