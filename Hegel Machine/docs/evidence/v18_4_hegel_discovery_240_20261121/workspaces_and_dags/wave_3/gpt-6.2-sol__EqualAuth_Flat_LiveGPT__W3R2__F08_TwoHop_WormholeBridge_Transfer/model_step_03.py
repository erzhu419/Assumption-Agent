import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s, root_u):
    a, b, c, d, k = p
    return (a + b / (1.0 + k*s)) * u / (d + u) + c*root_u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    u = np.maximum(u, 0.0)
    s = np.maximum(s, 0.0)
    root_u = np.sqrt(u)
    result = least_squares(lambda p: _law(p, u, s, root_u) - y, [2.0, 4.0, 0.4, 1.0, 1.0], bounds=([0.0, 0.0, 0.0, 1e-8, 0.0], np.inf))
    _params = result.x

def predict(row) -> float:
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return float(_law(_params, u, s, math.sqrt(u)))