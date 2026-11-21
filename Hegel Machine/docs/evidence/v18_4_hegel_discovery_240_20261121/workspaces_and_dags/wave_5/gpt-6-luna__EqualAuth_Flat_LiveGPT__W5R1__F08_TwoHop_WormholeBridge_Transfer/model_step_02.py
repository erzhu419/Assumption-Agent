import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    a, b, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return a * u / (b + u) + c * (s - 1.0)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([6.0, 1.0, -0.8]), bounds=([0.0, 1e-8, -20.0], [100.0, 100.0, 20.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _PARAMS))