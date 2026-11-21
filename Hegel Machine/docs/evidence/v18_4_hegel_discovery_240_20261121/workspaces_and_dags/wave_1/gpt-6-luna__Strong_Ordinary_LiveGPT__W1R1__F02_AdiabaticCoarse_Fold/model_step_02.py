import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    u = np.array([float(r['u_now']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * u + p[1] - y
    result = least_squares(residual, np.array([2.0, 0.0]))
    _PARAMS = result.x
    return {'slope': float(_PARAMS[0]), 'intercept': float(_PARAMS[1])}

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called first')
    return float(_PARAMS[0] * float(row['u_now']) + _PARAMS[1])