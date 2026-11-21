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
        a, b, c, k = p
        return a*u + (b*x + c*u)*np.exp(-k*h) - y
    best = None
    for k in (0.5, 1.0, 1.5, 2.0):
        result = least_squares(residual, [1.83, 1.0, -1.83, k], bounds=([-10, -100, -100, 0.05], [10, 100, 100, 4.0]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    a, b, c, k = _params
    return float(a*row['u_now'] + (b*row['x_now'] + c*row['u_now'])*math.exp(-k*row['H']))
