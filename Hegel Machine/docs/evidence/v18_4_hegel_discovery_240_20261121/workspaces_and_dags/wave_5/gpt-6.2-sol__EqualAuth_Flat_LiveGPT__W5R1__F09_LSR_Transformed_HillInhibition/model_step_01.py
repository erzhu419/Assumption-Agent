import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_tau = 1.0

def _law(u, s, p):
    vmax, k_half, alpha, leak = p
    z = u * u
    return vmax * z / ((k_half + z) * (1.0 + alpha * s)) - leak * u

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    u = u / _tau
    start = [2.0 * max(1.0, float(np.max(y))), 1.0, 1.0, 0.1]
    _params = least_squares(lambda p: _law(u, s, p) - y, start, bounds=([0.0, math.ulp(1.0), 0.0, 0.0], np.inf), x_scale='jac').x

def predict(row) -> float:
    return float(_law(float(row['u_now']) / _tau, float(row['s_grad']), _params))
