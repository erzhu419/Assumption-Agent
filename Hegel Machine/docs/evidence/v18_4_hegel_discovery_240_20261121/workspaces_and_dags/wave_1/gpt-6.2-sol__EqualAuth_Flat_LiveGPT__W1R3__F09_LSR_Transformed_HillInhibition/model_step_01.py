import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    z = u * u
    return vmax * (z / (k_half + z)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'] / 0.1, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _law(p, u, s) - y, [4.0, 1.0, 1.0, 0.1], bounds=([0.0, 1e-8, 0.0, 0.0], np.inf), x_scale='jac', gtol=1e-12).x

def predict(row) -> float:
    return float(_law(_params, row['u_now'] / 0.1, row['s_grad']))
