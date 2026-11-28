import numpy as np


def run_node(inputs: dict) -> dict:
    poly = inputs['node_01']['poly']
    kernel = inputs['node_02']
    smooth = kernel['smooth']

    # Correct moderate disagreement; fall back toward the smooth branch
    # when the polynomial prediction differs substantially.
    delta = poly[:, 0] - smooth[:, 0]
    y = smooth[:, 0] - 0.55 * delta / (1.0 + (delta / 0.06) ** 2)
    state = (0.4 * poly[:, 1:] + 0.6 * smooth[:, 1:]) @ kernel['vt']
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
