import numpy as np
from scipy.optimize import least_squares


def _basis(u, frequency):
    return np.column_stack((np.ones(len(u)), u, np.sin(u * frequency)))


def run_node(inputs: dict) -> dict:
    source = inputs['node_00']
    panel = source['panel_b']
    u, v = panel['u'], panel['v']

    def residual(frequency):
        design = _basis(u, frequency)
        weights = np.linalg.lstsq(design, v, rcond=None)[0]
        return (design @ weights - v).ravel()

    frequency = least_squares(residual, np.full(u.shape[1], 2.8),
                              bounds=(0.35, 6.0), max_nfev=180).x
    weights = np.linalg.lstsq(_basis(u, frequency), v, rcond=None)[0]
    return {key: source[key] for key in ('u_tr', 'u_te')} | {
        'b_tr': _basis(source['u_tr'], frequency) @ weights,
        'b_te': _basis(source['u_te'], frequency) @ weights,
    }
