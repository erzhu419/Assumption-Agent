import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([(r['u_now'], r['x_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    gain0 = float(np.dot(u, y) / np.dot(u, u))
    def residual(p):
        delta = -np.expm1(-p[1] * h)
        return x + (p[0] * u - x) * delta - y
    result = least_squares(residual, [gain0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    _params = tuple(map(float, result.x))
    return _params

def predict(row) -> float:
    gain, rate = _params
    x = row['x_now']
    return float(x - (gain * row['u_now'] - x) * math.expm1(-rate * row['H']))
