import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    un = u / 1.0
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un))
    return base + c_loc * math.sqrt(un)

def fit(train_rows):
    global _PARAMS
    def residual(p):
        return np.asarray([_value(r, p) - float(r['y']) for r in train_rows])
    result = least_squares(residual, x0=np.array([5.0, 0.7, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 1.5, 10.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))