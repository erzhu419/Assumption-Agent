import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    a, k, b, g, q = p
    return (a + g / (1.0 + q * s)) * u / (k + u) + b * np.sqrt(u)

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _law(u, s, p) - y, [2.0, 1.0, 0.4, 5.0, 1.0], bounds=([0.0, 1e-8, 0.0, 0.0, 0.0], np.inf)).x

def predict(row) -> float:
    a, k, b, g, q = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float((a + g / (1.0 + q * s)) * u / (k + u) + b * math.sqrt(u))