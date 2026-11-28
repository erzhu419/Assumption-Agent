"""Apply the learned endpoint and trajectory maps on the test design."""
import numpy as np


def run_node(inputs: dict) -> dict:
    design = inputs['node_00']
    fit = inputs['node_01']
    a, q = design['a_test'], design['q_test']
    z = a[:, 1:] @ fit['projection']
    features = np.column_stack((q, np.tanh(z)))
    y = features @ fit['readout']
    basis = fit['state_basis']
    state = (a @ fit['state_coef'] @ basis) @ basis.T
    final = state[:, -1]
    return {'y_hat': y, 'state_pred': state,
            'diagnostics_vec': np.array([y.mean(), y.std(), final.mean(), final.std()])}
