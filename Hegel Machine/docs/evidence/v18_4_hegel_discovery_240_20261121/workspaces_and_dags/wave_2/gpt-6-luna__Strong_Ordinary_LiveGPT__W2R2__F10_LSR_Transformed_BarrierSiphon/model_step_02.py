import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.asarray([_value(row, p) - float(row['y']) for row in train_rows])
    result = least_squares(residual, x0=np.array([5.0, 1.0, 0.5, 0.0]), bounds=([0.0, 0.0, 0.0, -20.0], [100.0, 20.0, 2.0, 20.0]), max_nfev=20000)
    _PARAMS = result.x
    return result

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))