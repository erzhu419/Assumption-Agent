import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, p):
    vmax, rho, c_loc = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)

def fit(train_rows):
    global _P
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    res = least_squares(residual, x0=np.array([5.0, 0.7, 0.3]), bounds=([0.0, 0.0, -10.0], [100.0, 1.0, 10.0]))
    _P = res.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_value(row, _P))