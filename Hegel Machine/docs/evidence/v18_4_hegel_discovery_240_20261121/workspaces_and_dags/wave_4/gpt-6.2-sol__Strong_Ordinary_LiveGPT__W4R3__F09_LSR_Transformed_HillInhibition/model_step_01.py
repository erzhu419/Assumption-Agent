import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u2, u, s):
    vmax, k_half, alpha, leak = p
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] / 0.1 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    u2 = u * u
    result = least_squares(lambda p: _law(p, u2, u, s) - y,
                           [10.0, 2.0, 0.5, 0.3],
                           bounds=([0.0, 1e-8, 0.0, 0.0], np.inf),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    u = row['u_now'] / 0.1
    return float(_law(_params, math.pow(u, 2), u, row['s_grad']))
