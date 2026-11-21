import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        k, a = p
        return np.array([r['y'] - (k*r['u_now'] + (r['x_now'] - k*r['u_now']) * math.exp(-a*r['H'])) for r in train_rows])
    res = least_squares(residual, x0=np.array([2.0, 0.2]), bounds=([-100.0, 1e-8], [100.0, 100.0]))
    _params = res.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    k, a = _params
    return float(k*row['u_now'] + (row['x_now'] - k*row['u_now']) * math.exp(-a*row['H']))