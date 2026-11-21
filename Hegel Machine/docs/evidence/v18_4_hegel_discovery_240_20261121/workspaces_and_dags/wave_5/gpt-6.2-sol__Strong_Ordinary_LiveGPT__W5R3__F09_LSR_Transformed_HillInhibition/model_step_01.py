import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(u, s, p):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * u2 / (k_half + u2) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'] / 0.1, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _response(u, s, p) - y, [8.0, 1.0, 1.0, 0.2], bounds=([0.0, 1e-9, 0.0, 0.0], np.inf)).x

def predict(row):
    return float(_response(row['u_now'] / 0.1, row['s_grad'], _params))