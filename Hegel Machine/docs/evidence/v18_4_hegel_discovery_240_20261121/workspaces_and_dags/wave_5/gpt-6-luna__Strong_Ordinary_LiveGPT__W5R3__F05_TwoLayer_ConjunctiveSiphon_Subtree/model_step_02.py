import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc = p
    u_n = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) + c_loc * math.sqrt(max(0.0, u_n))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    result = least_squares(residual, x0=np.array([5.0, 0.5, 0.0]), bounds=([0.0, 0.0, -20.0], [100.0, 2.0, 20.0]), max_nfev=20000)
    _PARAMS = result.x.copy()
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))