import numpy as np
from scipy.optimize import least_squares


def _basis(x, shape):
    squared = x * x
    return np.column_stack((np.ones(len(x)), x, x * (-np.expm1(-shape * squared))))


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_00']
    panel = upstream['panel_b']
    u, v = panel['u'], panel['v']

    def fit(shape):
        basis = _basis(u, shape)
        coefficients = np.linalg.lstsq(basis, v, rcond=None)[0]
        return basis @ coefficients - v, coefficients

    shape = least_squares(lambda k: fit(k)[0].ravel(), np.full(u.shape[1], 0.8),
                          bounds=(0.005, 5.0)).x
    _, coefficients = fit(shape)
    return {key: _basis(upstream[source], shape) @ coefficients
            for key, source in (('z_tr', 'cal_tr'), ('z_te', 'cal_te'))}
