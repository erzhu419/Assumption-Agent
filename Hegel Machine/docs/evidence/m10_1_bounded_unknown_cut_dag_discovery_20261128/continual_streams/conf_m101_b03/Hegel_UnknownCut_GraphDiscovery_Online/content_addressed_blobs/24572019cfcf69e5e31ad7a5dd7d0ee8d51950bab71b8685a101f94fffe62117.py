import numpy as np


def _design(x):
    # The panel readout mixes the four adjacent-coordinate products.
    adjacent_products = x * np.roll(x, -1, axis=1)
    return np.column_stack((np.ones(len(x)), adjacent_products))


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_00']
    panel = upstream['panel_c']
    coef = np.linalg.lstsq(_design(panel['u']), panel['v1'], rcond=None)[0]
    return {'a': _design(upstream['z']) @ coef}
