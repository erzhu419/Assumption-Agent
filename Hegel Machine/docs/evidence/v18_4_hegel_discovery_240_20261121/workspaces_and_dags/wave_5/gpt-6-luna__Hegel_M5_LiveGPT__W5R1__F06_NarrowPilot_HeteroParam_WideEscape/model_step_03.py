import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows], dtype=float) - y
    result = least_squares(residual, x0=np.array([6.0, 0.5, 0.0]), bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(row, _params))