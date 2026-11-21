import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha * s)) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in rows], dtype=float).T
    _params = least_squares(lambda p: _law(u, s, p) - y, [5.0, 1.0, 1.0, 0.2], bounds=([0.0, 1e-8, 0.0, 0.0], np.inf)).x

def predict(row) -> float:
    return float(_law(row['u_now'], row['s_grad'], _params))