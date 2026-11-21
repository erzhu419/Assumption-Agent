import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s, h):
    a, b, tau_fast, tau_slow, k = p
    fast = a * (u / (1.2 + u)) * (1.0 - k * s / (1.0 + s))
    return -fast * np.expm1(-h / tau_fast) + b * u * np.expm1(-h / tau_slow)

def fit(train_rows):
    global _params
    u, s, h, y = np.array([(r['u_now'], r['s_grad'], r['H'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _response(p, u, s, h) - y,
                           x0=[7.0, 2.0, 0.3, 11.0, 0.7],
                           bounds=([0.0, 0.0, 1e-6, 1e-6, 0.0],
                                   [math.inf, math.inf, math.inf, math.inf, 1.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_response(_params, row['u_now'], row['s_grad'], row['H']))