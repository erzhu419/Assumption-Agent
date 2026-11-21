import numpy as np
from scipy.optimize import least_squares
import math

_PARAMS = None

def _value(row, p):
    vmax, rho, c_sqrt = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _PARAMS
    ys = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([8.0, 0.5, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))