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
        a, k = p
        return a*u + (x-a*u)*np.exp(-k*h) - y
    result = least_squares(residual, [2.0, 1.3], bounds=([0.0, 0.0], [np.inf, np.inf]))
    _params = result.x

def predict(row):
    a, k = _params
    u = float(row['u_now'])
    x = float(row['x_now'])
    h = float(row['H'])
    return float(a*u + (x-a*u)*np.exp(-k*h))