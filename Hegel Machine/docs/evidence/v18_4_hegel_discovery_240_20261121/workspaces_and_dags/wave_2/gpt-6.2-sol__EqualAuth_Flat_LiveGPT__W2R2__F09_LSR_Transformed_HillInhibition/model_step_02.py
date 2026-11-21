import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_tau = 1.0

def _law(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha_inh * s)) - c_leak * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'] / _tau, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _law(p, u, s) - y, [6.0, 1.0, 1.0, 0.1], bounds=([0.0, 1e-10, 0.0, 0.0], np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    return float(_law(_params, row['u_now'] / _tau, row['s_grad']))