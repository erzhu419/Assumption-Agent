import numpy as np


def _basis(x):
    return np.column_stack((np.ones(len(x)), x, np.cos(2.4 * x)))


def run_node(inputs: dict) -> dict:
    panel = inputs['node_00']
    coefficients = np.linalg.lstsq(_basis(panel['u']), panel['v2'], rcond=None)[0]
    return {'d': _basis(panel['x']) @ coefficients}
