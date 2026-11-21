import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([[r[k] for k in ('u_now', 'x_now', 'H', 'y')] for r in train_rows], dtype=float)
    u, x, h, y = data.T
    def residual(p):
        k, a = p
        return x * np.exp(-a * h) - k * u * np.expm1(-a * h) - y
    k0 = max(float(np.dot(u, y) / np.dot(u, u)), 1e-6)
    result = least_squares(residual, [k0, 1.0], bounds=(0.0, np.inf))
    _params = tuple(map(float, result.x))
    return _params

def predict(row):
    k, a = _params
    t = -a * float(row['H'])
    return float(row['x_now']) * math.exp(t) - k * float(row['u_now']) * math.expm1(t)
