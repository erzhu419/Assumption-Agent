import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    s = np.array([float(r['s_grad']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def pred(p):
        vmax, rho, c_sqrt = p
        return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_sqrt * np.sqrt(np.maximum(u, 0.0))
    result = least_squares(lambda p: pred(p) - y, x0=np.array([8.0, 0.7, 0.5]), bounds=([0.0, 0.0, -10.0], [100.0, 1.5, 10.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    vmax, rho, c_sqrt = _PARAMS
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_sqrt * math.sqrt(u))