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
    result = least_squares(residual, [2.05, 1.0, -2.05, 1.2], bounds=([-10, -20, -20, 0], [10, 20, 20, 5]), max_nfev=5000)
    _params = result.x

def predict(row):
    a, b, c, k = _params
    return float(a*row['u_now'] + (b*row['x_now'] + c*row['u_now'])*math.exp(-k*row['H']))