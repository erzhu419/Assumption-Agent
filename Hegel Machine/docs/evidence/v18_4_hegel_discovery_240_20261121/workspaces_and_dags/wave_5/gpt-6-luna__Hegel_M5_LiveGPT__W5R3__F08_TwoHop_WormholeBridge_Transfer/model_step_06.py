import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    un = u  # characteristic tau scale is 1.0
    return vmax * (1.0 - rho * s / (1.0 + s)) * (un / (1.2 + un)) + c_loc * math.sqrt(max(un, 0.0))

def fit(train_rows):
    global _P
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows], dtype=float) - y
    result = least_squares(residual, x0=np.array([6.0, 0.7, 0.1]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    _P = result.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _P))