import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.0, 0.7, 0.0]), bounds=([0.0, 0.0, -10.0], [20.0, 1.5, 10.0]), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise ValueError('Call fit(train_rows) first.')
    return float(_value(row, _params))