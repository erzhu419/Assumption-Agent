import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u / 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_sqrt * math.sqrt(max(0.0, un))

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([4.0, 0.7, 0.0]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise ValueError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _PARAMS))