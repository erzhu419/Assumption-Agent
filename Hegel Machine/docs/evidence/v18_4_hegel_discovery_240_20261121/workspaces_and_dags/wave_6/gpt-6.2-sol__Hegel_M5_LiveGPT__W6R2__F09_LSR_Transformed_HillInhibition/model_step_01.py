import math
import numpy as np
from scipy.optimize import least_squares

TAU = 0.1
_params = None

def _law(p, u, s):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) / TAU for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    result = least_squares(lambda p: _law(p, u, s) - y,
                           [max(1.0, 2.0 * float(np.max(y))), 1.0, 0.5, 0.1],
                           bounds=([0.0, math.ulp(1.0), 0.0, 0.0], np.inf),
                           x_scale='jac')
    _params = result.x
    return _params

def predict(row):
    return float(_law(_params, float(row['u_now']) / TAU, float(row['s_grad'])))