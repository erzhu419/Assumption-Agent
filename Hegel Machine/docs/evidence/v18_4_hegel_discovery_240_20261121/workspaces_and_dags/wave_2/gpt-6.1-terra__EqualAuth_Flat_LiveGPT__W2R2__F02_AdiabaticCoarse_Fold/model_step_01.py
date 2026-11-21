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
    solutions = [least_squares(residual, [k, a], bounds=([0.0, 0.0], [np.inf, np.inf])) for k in (1.0, 2.0, 3.0) for a in (0.1, 1.0, 3.0)]
    _params = min(solutions, key=lambda s: np.sum(s.fun**2)).x

def predict(row):
    k, a = _params
    steady = k * float(row['u_now'])
    return float(steady + (float(row['x_now']) - steady) * math.exp(-a * float(row['H'])))
