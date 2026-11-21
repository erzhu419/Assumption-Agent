import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([[r[key] for key in ('u_now', 'x_now', 'H', 'y')] for r in train_rows], dtype=float).T
    k0 = np.dot(u, y) / np.dot(u, u)
    def residual(p):
        k, a = p
        eq = k * u
        return eq + (x - eq) * np.exp(-a * h) - y
    _params = least_squares(residual, [k0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    k, a = _params
    eq = k * row['u_now']
    return float(eq + (row['x_now'] - eq) * math.exp(-a * row['H']))
