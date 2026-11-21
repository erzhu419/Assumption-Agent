import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        k, a = p
        return k*u + (x-k*u)*np.exp(-a*h) - y
    k0 = float(np.dot(u, y) / np.dot(u, u))
    fits = [least_squares(residual, [k0, a0], bounds=([-np.inf, 0.0], [np.inf, np.inf])) for a0 in (0.1, 0.5, 1.0, 2.0)]
    _params = min(fits, key=lambda f: np.dot(f.fun, f.fun)).x

def predict(row):
    k, a = _params
    ss = k * float(row['u_now'])
    return float(ss + (float(row['x_now']) - ss) * math.exp(-a * float(row['H'])))