import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU = 0.02

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u_n = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return j0 * u_n * math.exp(-beta_bar / (0.5 + u_n)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u_n, 0.0))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([0.5, 1.0, 0.3, 0.05]), bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))