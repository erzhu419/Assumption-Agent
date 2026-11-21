import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    u_factor = u / (1.2 + u)
    inhibition = 1.0 - rho * s / (1.0 + s)
    return vmax * inhibition * u_factor + c_loc * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([7.0, 0.7, 0.0]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    _PARAMS = result.x
    return tuple(float(v) for v in _PARAMS)

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))