import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.array([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    def residual(p):
        k, a = p
        x_ss = k * u
        return x_ss + (x - x_ss) * np.exp(-a * h) - y
    k0 = float(np.dot(u, y) / np.dot(u, u))
    _params = least_squares(residual, [max(k0, 1e-8), 1.0], bounds=(0.0, np.inf), gtol=1e-12).x

def predict(row) -> float:
    k, a = _params
    x_ss = k * row['u_now']
    return float(x_ss + (row['x_now'] - x_ss) * math.exp(-a * row['H']))
