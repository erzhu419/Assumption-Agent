import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        k, a = p
        return x * np.exp(-a * h) - k * u * np.expm1(-a * h) - y
    _params = least_squares(residual, [np.dot(u, y) / np.dot(u, u), 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x

def predict(row):
    k, a = _params
    z = -a * row['H']
    return float(row['x_now'] * math.exp(z) - k * row['u_now'] * math.expm1(z))
