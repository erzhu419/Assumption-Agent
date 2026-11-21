import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = max(float(row['u_now']), 0.0)
    s = float(row['s_grad'])
    return u, s

def _law(row, p):
    a, k, c = p
    u, s = _features(row)
    return a * u / (1.0 + u / k) * math.exp(-c * (s - 1.0))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_law(r, p) for r in train_rows]) - y
    result = least_squares(residual, x0=np.array([3.2, 1.5, 0.35]), bounds=([0.001, 0.001, -5.0], [100.0, 100.0, 5.0]), max_nfev=20000)
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_law(row, _PARAMS))