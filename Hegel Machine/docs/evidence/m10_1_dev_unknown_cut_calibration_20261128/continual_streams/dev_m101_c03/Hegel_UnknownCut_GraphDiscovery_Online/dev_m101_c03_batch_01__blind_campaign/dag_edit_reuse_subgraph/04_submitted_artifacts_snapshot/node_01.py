import numpy as np


def _basis(x):
    # The v1 readout is affine in the products between {0, 2} and {1, 3}.
    x0, x1, x2, x3 = x.T
    return np.column_stack((np.ones(len(x)), x0*x1, x0*x3, x2*x1, x2*x3))


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_00']
    coefficients = np.linalg.lstsq(_basis(upstream['u']), upstream['v1'], rcond=None)[0]
    return {'a': _basis(upstream['z']) @ coefficients}
