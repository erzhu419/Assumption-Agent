import numpy as np
from scipy.optimize import least_squares
import math

_PARAMS = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u  # characteristic tau scale is 1.0
    return (j0 * u_n * math.exp(-beta_bar / (0.5 + u_n))
            * (1.0 - rho * s / (1.0 + s))
            + c_sqrt * math.sqrt(u_n))

def fit(train_rows):
    global _PARAMS
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.5, 0.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _PARAMS))