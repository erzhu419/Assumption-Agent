import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _P
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 0.5, 0.5]), bounds=([0.0, 0.0, -10.0], [20.0, 2.0, 10.0]))
    _P = result.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _P))