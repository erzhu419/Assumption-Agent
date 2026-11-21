import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(u, s, p):
    a, b, c, k, e, h, j = p
    quadratic = k * u * u
    base = 1.0 + c * u + quadratic
    load = e * s / (1.0 + h * s)
    return u * (a + b * u) / (base + load * (1.0 + j * u + quadratic))

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _response(u, s, p) - y,
                           [5.0, 3.5, 1.4, 0.6, 1.0, 0.2, 0.8],
                           bounds=(0.0, np.inf), x_scale='jac')
    _params = result.x

def predict(row) -> float:
    return float(_response(float(row['u_now']), float(row['s_grad']), _params))
