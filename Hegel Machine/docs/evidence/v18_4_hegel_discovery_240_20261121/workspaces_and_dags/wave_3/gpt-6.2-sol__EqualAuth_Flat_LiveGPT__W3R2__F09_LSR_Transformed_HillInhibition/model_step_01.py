import math
import numpy as np
from scipy.optimize import least_squares

TAU = 1.0

def _law(p, u, s):
    vmax, k_half, alpha, leak = p
    q = u * u
    return vmax * q / (k_half + q) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'] / TAU, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _law(p, u, s) - y, [8.0, 1.0, 0.5, 0.2], bounds=([0.0, 1e-12, 0.0, 0.0], math.inf)).x
    return _params

def predict(row):
    return float(_law(_params, row['u_now'] / TAU, row['s_grad']))