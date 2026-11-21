import math
import numpy as np
from scipy.optimize import least_squares

_tau = 1.0
_params = None

def _law(p, u, s):
    vmax, k_half, alpha, leak = p
    q = u * u
    return vmax * (q / (k_half + q)) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'] / _tau, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    scale = max(1.0, float(np.max(np.abs(y))))
    result = least_squares(lambda p: (_law(p, u, s) - y) / scale,
                           [2.0 * scale, 1.0, 1.0, 0.1],
                           bounds=([0.0, 1e-9, 0.0, 0.0], np.inf),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    return float(_law(_params, row['u_now'] / _tau, row['s_grad']))