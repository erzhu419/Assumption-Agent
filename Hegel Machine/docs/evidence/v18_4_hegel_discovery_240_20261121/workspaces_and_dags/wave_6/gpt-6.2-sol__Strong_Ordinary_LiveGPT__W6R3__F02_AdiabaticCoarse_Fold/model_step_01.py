import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.array([[r[k] for k in ('u_now', 'x_now', 'H', 'y')] for r in train_rows], dtype=float).T
    gain0 = np.dot(u, y) / np.dot(u, u)
    def residual(p):
        eq = p[0] * u
        return eq + (x - eq) * np.exp(-p[1] * h) - y
    _params = least_squares(residual, [gain0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x

def predict(row):
    gain, rate = _params
    eq = gain * row['u_now']
    return float(eq + (row['x_now'] - eq) * math.exp(-rate * row['H']))