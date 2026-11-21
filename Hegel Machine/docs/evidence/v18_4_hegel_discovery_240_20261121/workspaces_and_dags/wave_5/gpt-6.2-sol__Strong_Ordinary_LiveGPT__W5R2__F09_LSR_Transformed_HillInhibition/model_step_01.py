import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha * s)) - leak * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'] / 2.5, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _law(u, s, p) - y, [2.0 * max(y), 1.0, 0.5, 0.2], bounds=([0.0, 1e-10, 0.0, 0.0], np.inf))
    _params = result.x

def predict(row) -> float:
    return float(_law(row['u_now'] / 2.5, row['s_grad'], _params))