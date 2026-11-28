"""Apply the fitted readouts to the calibrated test design."""
import numpy as np


def run_node(inputs: dict) -> dict:
    raw, fitted = inputs['node_00'], inputs['node_01']
    z = fitted['test_features']
    if len(z) != len(raw['a_test']):
        raise ValueError('Calibrated design must retain all test rows')
    values = z @ fitted['coefficients']
    endpoint, state = values[:, 0], values[:, 1:]
    last = state[:, -1]
    return {'y_hat': endpoint, 'state_pred': state,
            'diagnostics_vec': np.array([endpoint.mean(), endpoint.std(), last.mean(), last.std()])}
