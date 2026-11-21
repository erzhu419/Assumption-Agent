import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    u = np.array([float(r['u_raw']) for r in rows])
    m = np.array([float(r['m_2']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        vmax, k_m, c_raw = p
        return vmax * m / (k_m + m) + c_raw * np.sqrt(u) - y
    res = least_squares(residual, x0=[8.0, 1.0, 0.1], bounds=([0.0, 1e-9, -np.inf], [np.inf, np.inf, np.inf]))
    _PARAMS = tuple(float(v) for v in res.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    vmax, k_m, c_raw = _PARAMS
    u = float(row['u_raw'])
    m = float(row['m_2'])
    return float(vmax * m / (k_m + m) + c_raw * math.sqrt(u))