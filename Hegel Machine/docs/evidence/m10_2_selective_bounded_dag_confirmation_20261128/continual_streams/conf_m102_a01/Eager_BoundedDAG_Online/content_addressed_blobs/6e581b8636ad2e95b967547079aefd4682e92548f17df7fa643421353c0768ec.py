import numpy as np


def run_node(inputs: dict) -> dict:
    poly = inputs['node_01']['poly']
    kernel = inputs['node_02']
    smooth = kernel['smooth']

    # Correct scalar bias using the disagreement between the two branches.
    y = smooth[:, 0] + 0.5 * (smooth[:, 0] - poly[:, 0])
    state = (0.4 * poly[:, 1:] + 0.6 * smooth[:, 1:]) @ kernel['vt']
    last = state[:, -1]
    return {
        'y_hat': y,
        'state_pred': state,
        'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()]),
    }
