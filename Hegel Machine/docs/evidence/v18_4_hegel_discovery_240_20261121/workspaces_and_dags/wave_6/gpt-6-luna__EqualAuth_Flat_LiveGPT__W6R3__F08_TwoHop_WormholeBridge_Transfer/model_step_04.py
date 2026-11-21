import math
import numpy as np
from scipy.optimize import least_squares

_P = None

def _predict_params(p, row):
    a, b, c, d, e = p
    u = float(row['u_now'])
    g = float(row['s_grad'])
    q = u / (c + u)
    return a + b*q + (g - 1.0)*(d*q + e*u)

def fit(train_rows):
    global _P
    ys = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_predict_params(p, r) for r in train_rows]) - ys
    result = least_squares(residual, x0=np.array([0.3, 2.5, 0.8, -1.0, 0.1]), bounds=([-10.0, -10.0, 0.01, -10.0, -10.0], [10.0, 10.0, 20.0, 10.0, 10.0]), max_nfev=20000)
    _P = result.x
    return _P

def predict(row) -> float:
    if _P is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(_P, row))