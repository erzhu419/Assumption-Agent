import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, g, root_u, p):
    a, k, b, c = p
    return (a + b / (1.0 + g)) * u / (k + u) + c * root_u

def fit(train_rows):
    global _params
    u, g, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root_u = np.sqrt(u)
    _params = least_squares(lambda p: _law(u, g, root_u, p) - y, [2.4, 1.2, 5.8, 0.33], bounds=([0.0, 1e-8, 0.0, 0.0], np.inf)).x

def predict(row) -> float:
    u = float(row['u_now'])
    return float(_law(u, float(row['s_grad']), math.sqrt(u), _params))
