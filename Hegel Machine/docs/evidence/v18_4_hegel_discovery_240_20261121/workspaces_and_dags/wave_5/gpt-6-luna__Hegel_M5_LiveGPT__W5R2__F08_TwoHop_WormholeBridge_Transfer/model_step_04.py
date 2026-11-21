import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    def features(r):
        u = float(r['u_now'])
        g = float(r['s_grad'])
        return np.array([u, u**4, g, g*u], dtype=float)
    X = np.vstack([features(r) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    result = least_squares(lambda p: X @ p - y, np.zeros(4), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    u = float(row['u_now'])
    g = float(row['s_grad'])
    a, b, c, d = _PARAMS
    return float(a*u + b*u**4 + c*g + d*g*u)