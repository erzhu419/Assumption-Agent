import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        k, a = p
        return -k * u * np.expm1(-a * h) + x * np.exp(-a * h) - y
    k0 = float(np.dot(u, y) / np.dot(u, u))
    _params = tuple(least_squares(residual, [max(k0, 0.0), 1.0], bounds=(0.0, np.inf)).x)

def predict(row) -> float:
    k, a = _params
    z = -a * row['H']
    return float(-k * row['u_now'] * math.expm1(z) + row['x_now'] * math.exp(z))
