import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(max(0.0, un))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.5, 0.5]), bounds=([0.0, -5.0, -10.0], [100.0, 5.0, 10.0]))
    _params = result.x
    return _params.copy()

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))