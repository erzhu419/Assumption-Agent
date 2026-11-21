import math
import numpy as np
from scipy.optimize import least_squares

TAU = 0.1
_params = None

def _law(p, u, s):
    vmax, k_half, alpha, leak = p
    q = u * u
    return vmax * q / ((k_half + q) * (1.0 + alpha * s)) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.asarray([r['u_now'] / TAU for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    y = np.asarray([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _law(p, u, s) - y,
                           [8.0, 2.0, 1.0, 0.3],
                           bounds=([0.0, 1e-12, 0.0, 0.0], np.inf),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_law(_params, row['u_now'] / TAU, row['s_grad']))
