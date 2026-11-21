import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha_inh * s)) - c_leak * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'] / 0.1, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _law(u, s, p) - y, x0=[2.0 * max(1.0, float(np.max(y))), 1.0, 1.0, 0.2], bounds=([0.0, 1e-9, 0.0, 0.0], np.inf))
    _params = result.x

def predict(row) -> float:
    return float(_law(row['u_now'] / 0.1, row['s_grad'], _params))