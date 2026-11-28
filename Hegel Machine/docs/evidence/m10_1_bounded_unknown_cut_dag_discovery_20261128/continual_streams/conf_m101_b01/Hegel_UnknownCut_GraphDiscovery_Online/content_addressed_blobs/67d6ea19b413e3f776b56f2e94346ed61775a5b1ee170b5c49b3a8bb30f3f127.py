import numpy as np
from scipy.optimize import least_squares


def _design(x, params):
    rate, amplitude, power = params
    warped = x * (1.0 + amplitude * np.exp(-rate * np.abs(x)**power))
    return np.column_stack((np.ones(len(x)), warped))


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_00']
    panel = upstream['panel_b']
    u, v = panel['u'], panel['v']

    def project(params):
        design = _design(u, params)
        weights = np.linalg.lstsq(design, v, rcond=None)[0]
        return design, weights

    def residual(params):
        design, weights = project(params)
        return (design @ weights - v).ravel()

    params = least_squares(
        residual, [1.0, -1.1, 2.0],
        bounds=([0.1, -2.0, 0.7], [6.0, -0.1, 5.0]), ftol=1e-11,
    ).x
    _, weights = project(params)
    return {'x': _design(upstream['calibrated_x'], params) @ weights}
