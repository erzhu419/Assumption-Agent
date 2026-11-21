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
        e = np.exp(-a * h)
        return k * u + (x - k * u) * e - y
    k0 = float(np.dot(u, y) / max(np.dot(u, u), 1e-12))
    fits = [least_squares(residual, [k0, a0], bounds=([-np.inf, 0.0], [np.inf, np.inf])) for a0 in (0.2, 0.8, 2.0)]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    k, a = _params
    steady = k * float(row['u_now'])
    return float(steady + (float(row['x_now']) - steady) * math.exp(-a * float(row['H'])))
