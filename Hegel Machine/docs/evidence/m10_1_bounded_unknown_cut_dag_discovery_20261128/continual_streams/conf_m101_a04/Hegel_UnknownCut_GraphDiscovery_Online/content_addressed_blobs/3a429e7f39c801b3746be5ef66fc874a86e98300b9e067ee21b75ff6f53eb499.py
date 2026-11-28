import numpy as np


def _basis(x):
    return np.column_stack((np.ones(len(x)), *(x[:, i]*x[:, j] for i in range(x.shape[1]) for j in range(i + 1, x.shape[1]))))


def run_node(inputs: dict) -> dict:
    panel = inputs['node_00']
    coef = np.linalg.lstsq(_basis(panel['u']), panel['v1'], rcond=None)[0]
    return {'b': _basis(panel['x']) @ coef}
