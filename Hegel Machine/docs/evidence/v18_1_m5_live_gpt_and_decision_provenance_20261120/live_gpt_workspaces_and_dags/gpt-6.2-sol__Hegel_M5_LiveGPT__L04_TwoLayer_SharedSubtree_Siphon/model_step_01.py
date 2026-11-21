import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now']) / 1.0
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def _law(p, a, b, c):
    vmax, rho, c_loc = p
    return vmax * (1.0 - rho * b) * a + c_loc * c

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    a, b, c = np.asarray([_features(row) for row in rows], dtype=float).T
    y = np.asarray([float(row['y']) for row in rows])
    result = least_squares(lambda p: _law(p, a, b, c) - y,
                           x0=[6.5, 0.70, 0.0],
                           bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    return float(_law(_PARAMS, *_features(row)))