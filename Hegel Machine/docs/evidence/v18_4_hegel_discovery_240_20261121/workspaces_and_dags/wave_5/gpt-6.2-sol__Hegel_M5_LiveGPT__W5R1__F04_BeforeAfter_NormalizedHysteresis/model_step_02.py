import math
import numpy as np
from scipy.optimize import least_squares

_SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'] / _SCALE, (r['u_now'] - r['u_past']) / _SCALE, r['H'], r['y']) for r in train_rows], dtype=float)
    u, step, h, y = data.T
    def residual(p):
        c, a, k = p
        return c * u + a * step * np.exp(-k * h) - y
    _params = least_squares(residual, [1.0, 1.0, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    c, a, k = _params
    u = row['u_now'] / _SCALE
    step = (row['u_now'] - row['u_past']) / _SCALE
    return float(c * u + a * step * math.exp(-k * row['H']))
