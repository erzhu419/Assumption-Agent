import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([(r['u_now'], r['x_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        k, a = p
        q = -a * h
        return x * np.exp(q) - k * u * np.expm1(q) - y
    k0 = float(np.dot(u, y) / np.dot(u, u))
    result = least_squares(residual, [k0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    _params = tuple(map(float, result.x))

def predict(row) -> float:
    k, a = _params
    q = -a * float(row['H'])
    return float(float(row['x_now']) * math.exp(q) - k * float(row['u_now']) * math.expm1(q))