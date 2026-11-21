import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(u, s, h, p, expm1):
    a, tau_fast, b, tau_slow, k = p
    fast = a * u / (1.2 + u) * (1.0 - k * s / (1.0 + s))
    return -fast * expm1(-h / tau_fast) + b * u * expm1(-h / tau_slow)

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([(r['u_now'], r['s_grad'], r['H'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(
        lambda p: _response(u, s, h, p, np.expm1) - y,
        [5.0, 0.5, 2.0, 10.0, 0.5],
        bounds=([0.0, 1e-4, 0.0, 1e-4, 0.0], [np.inf, np.inf, np.inf, np.inf, 1.0]),
        x_scale='jac',
    )
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_response(row['u_now'], row['s_grad'], row['H'], _params, math.expm1))