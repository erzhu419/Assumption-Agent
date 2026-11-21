import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def pred(p):
        k, a = p
        z = np.exp(-a * H)
        return k*u + (x-k*u)*z
    result = least_squares(lambda p: pred(p)-y, x0=[2.5, 1.0], bounds=([0.0, 0.0], [20.0, 100.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    k, a = _PARAMS
    z = math.exp(-a * row['H'])
    return float(k*row['u_now'] + (row['x_now']-k*row['u_now'])*z)
