import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, leak_rate = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    u_n = u / 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u_n / (1.2 + u_n)) - leak_rate * u_n

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.5, 0.7, 0.1]), bounds=([0.0, 0.0, 0.0], [20.0, 2.0, 10.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))