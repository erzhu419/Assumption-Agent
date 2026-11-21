import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _value(row, p):
    u = max(1e-8, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    a, k, c, b = p
    base = a * u / (k + u) + c * math.sqrt(u)
    return base * math.exp(-b * s)

def fit(train_rows):
    global _P
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=np.array([4.0, 1.0, 0.5, 0.3]), bounds=([0.0, 1e-5, 0.0, 0.0], [100.0, 100.0, 100.0, 5.0]), max_nfev=20000)
    _P = result.x.copy()
    return _P

def predict(row):
    if _P is None:
        raise ValueError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _P))