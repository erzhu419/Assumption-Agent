import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = np.array([3.0, 3.0, 1.5], dtype=float)

def _predict_with(params, row):
    a, b, k = params
    u1 = float(row['u_1'])
    u2 = float(row['u_2'])
    return a * math.log1p(u1) + b * u2 / (k + u2)

def fit(train_rows):
    global _PARAMS
    x = np.array([[float(r['u_1']), float(r['u_2'])] for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        a, b, k = p
        pred = a * np.log1p(x[:, 0]) + b * x[:, 1] / (k + x[:, 1])
        return pred - y
    result = least_squares(residual, _PARAMS, bounds=([-20.0, -20.0, 0.01], [20.0, 20.0, 50.0]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    return float(_predict_with(_PARAMS, row))