"""Fit an origin-centered tanh endpoint readout and the shared rank-two trajectory."""
import numpy as np
from scipy.optimize import least_squares


def _fit_readout(a, q, y):
    x = a[:, 1:]

    def solve(p):
        features = np.column_stack((q, np.tanh(x @ p)))
        coef = np.linalg.lstsq(features, y, rcond=1e-9)[0]
        return features @ coef - y, coef

    # Profile out the quadratic and saturation coefficients. Keep the
    # nonlinear correction odd under reversal of the projected coordinate.
    w = np.linalg.lstsq(a, y, rcond=None)[0][1:]
    norm = np.linalg.norm(w)
    if norm < 1e-10:
        p = np.zeros(x.shape[1])
        return p, solve(p)[1]
    direction = w / norm
    best = min(
        (least_squares(lambda p: solve(p)[0], scale * direction,
                       bounds=(-5.0, 5.0), max_nfev=65)
         for scale in (0.55, 1.25, 2.2, 3.4)),
        key=lambda result: result.cost,
    )
    return best.x, solve(best.x)[1]


def run_node(inputs: dict) -> dict:
    train = inputs['__input__']['train']
    design = inputs['node_00']
    a, q = design['a'], design['q']
    projection, readout = _fit_readout(a, q, np.asarray(train['y'], dtype=float))
    state_coef = np.linalg.lstsq(a, train['traj'], rcond=None)[0]
    _, _, modes = np.linalg.svd(a @ state_coef, full_matrices=False)
    basis = modes[:2].T
    return {'projection': projection, 'readout': readout,
            'state_coef': state_coef, 'state_basis': basis}
