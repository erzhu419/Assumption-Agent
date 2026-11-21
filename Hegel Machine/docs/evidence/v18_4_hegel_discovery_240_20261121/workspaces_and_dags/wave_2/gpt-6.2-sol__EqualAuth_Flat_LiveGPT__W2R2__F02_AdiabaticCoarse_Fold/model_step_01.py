import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.array([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        k, a = p
        return x + (k*u - x)*(-np.expm1(-a*h)) - y
    k0 = float(np.dot(u, y) / np.dot(u, u))
    _params = least_squares(residual, [k0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x

def predict(row) -> float:
    k, a = _params
    x = row['x_now']
    return float(x + (k*row['u_now'] - x)*(-math.expm1(-a*row['H'])))
