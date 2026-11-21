import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([7.0, 0.7, 0.5]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _PARAMS))