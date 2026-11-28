import numpy as np
from scipy.optimize import minimize_scalar


def _basis(x, frequency):
    return np.column_stack((np.ones(len(x)), np.cos(frequency * x)))


def _fit(u, v2):
    def solve(frequency):
        design = _basis(u, frequency)
        coefficients = np.linalg.lstsq(design, v2, rcond=None)[0]
        residual = design @ coefficients - v2
        return np.sum(residual**2), coefficients

    # All four coordinate-wise cosine features share one frequency.
    optimum = minimize_scalar(lambda f: solve(f)[0], bounds=(1.5, 3.5),
                              method='bounded', options={'xatol': 1e-10})
    return optimum.x, solve(optimum.x)[1]


def run_node(inputs: dict) -> dict:
    upstream = inputs['node_00']
    frequency, coefficients = _fit(upstream['u'], upstream['v2'])
    return {'b': _basis(upstream['z'], frequency) @ coefficients}
