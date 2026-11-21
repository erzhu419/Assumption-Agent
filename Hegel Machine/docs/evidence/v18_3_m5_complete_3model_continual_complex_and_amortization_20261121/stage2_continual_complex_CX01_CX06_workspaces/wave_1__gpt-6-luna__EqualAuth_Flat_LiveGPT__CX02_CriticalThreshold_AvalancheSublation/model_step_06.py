import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic_tau_scale = 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) - leak * un

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([8.0, 0.7, 0.5]), bounds=([0.0, 0.0, 0.0], [100.0, 5.0, 10.0]))
    _PARAMS = result.x
    return tuple(float(v) for v in _PARAMS)

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))