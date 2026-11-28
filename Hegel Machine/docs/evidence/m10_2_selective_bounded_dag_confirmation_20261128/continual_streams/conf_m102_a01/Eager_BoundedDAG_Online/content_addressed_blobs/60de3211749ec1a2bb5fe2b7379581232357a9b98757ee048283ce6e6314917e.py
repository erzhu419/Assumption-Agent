import numpy as np


def run_node(inputs: dict) -> dict:
    poly = inputs['node_01']['poly']
    branch = inputs['node_02']
    smooth, basis = branch['smooth'], branch['vt']
    state = (.4 * poly[:, 1:] + .6 * smooth[:, 1:]) @ basis

    # Remove the leading endpoint trend from the trajectory disagreement.
    gap = (poly[:, 1:] - smooth[:, 1:]) @ basis
    shape_gap = np.clip((gap[:, 0] - .2 * gap[:, -1]) / .008, -1., 1.)
    s = smooth[:, 0]
    bounded_s = np.clip(s, -1., 1.)
    y = (s - .15 * (poly[:, 0] - s)
         + .0375 * bounded_s**3 + .025 * bounded_s * shape_gap)
    last = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), last.mean(), last.std()])}
