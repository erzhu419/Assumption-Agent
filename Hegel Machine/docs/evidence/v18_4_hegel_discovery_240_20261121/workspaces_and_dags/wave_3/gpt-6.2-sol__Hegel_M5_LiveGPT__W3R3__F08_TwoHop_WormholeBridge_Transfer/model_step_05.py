import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s):
    a, k, n, c, b, d = p
    q = np.power(np.maximum(u, 0.0), n)
    return q / (k + q) * (a / (1.0 + c * s) + b + d / (1.0 + np.maximum(u, 0.0)))

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _response(p, u, s) - y,
                            [4.5, 1.0, 1.1, 1.0, 1.3, 0.3],
                            bounds=([0.0, 1e-8, 0.1, 0.0, 0.0, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    u = max(0.0, float(row['u_now']))
    q = math.pow(u, _params[2])
    a, k, _, c, b, d = _params
    return float(q / (k + q) * (a / (1.0 + c * row['s_grad']) + b + d / (1.0 + u)))