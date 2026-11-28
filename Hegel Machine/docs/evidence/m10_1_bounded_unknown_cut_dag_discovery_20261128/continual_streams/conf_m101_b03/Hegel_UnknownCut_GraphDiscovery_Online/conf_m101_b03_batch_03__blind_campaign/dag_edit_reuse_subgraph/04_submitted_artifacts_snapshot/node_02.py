import numpy as np
from scipy.optimize import minimize_scalar


def _design(x, frequency):
    return np.column_stack((np.ones(len(x)), np.cos(frequency * x)))


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_00']
    panel = upstream['panel_c']
    u, v = panel['u'], panel['v2']

    # All four coordinates share a cosine frequency; the mixing is affine.
    def residual_sum_squares(frequency):
        design = _design(u, frequency)
        coef = np.linalg.lstsq(design, v, rcond=None)[0]
        residual = design @ coef - v
        return np.sum(residual**2)

    frequency = minimize_scalar(residual_sum_squares, bounds=(1.0, 4.0),
                                method='bounded').x
    coef = np.linalg.lstsq(_design(u, frequency), v, rcond=None)[0]
    return {'b': _design(upstream['z'], frequency) @ coef}
