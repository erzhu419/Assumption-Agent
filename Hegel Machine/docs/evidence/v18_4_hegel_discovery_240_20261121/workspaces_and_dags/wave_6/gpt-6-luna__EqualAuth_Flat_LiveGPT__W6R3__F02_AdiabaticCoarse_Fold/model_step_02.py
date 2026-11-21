import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([float(r['u_now']) for r in train_rows], dtype=float)
    x = np.asarray([float(r['x_now']) for r in train_rows], dtype=float)
    H = np.asarray([float(r['H']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        k, a = p
        return k*u + (x-k*u)*np.exp(-a*H) - y
    result = least_squares(residual, x0=np.array([2.38, 5.0]), bounds=([0.0, 0.0], [20.0, 100.0]))
    _params = (float(result.x[0]), float(result.x[1]))
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    k, a = _params
    u = float(row['u_now'])
    x = float(row['x_now'])
    H = float(row['H'])
    return float(k*u + (x-k*u)*math.exp(-a*H))