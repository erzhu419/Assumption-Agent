import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    u, x, h, y = data.T
    def residual(p):
        k, a = p
        return x + (k * u - x) * (-np.expm1(-a * h)) - y
    _params = least_squares(residual, [np.dot(u, y) / np.dot(u, u), 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x

def predict(row):
    k, a = _params
    x = row['x_now']
    return float(x + (k * row['u_now'] - x) * (-math.expm1(-a * row['H'])))
