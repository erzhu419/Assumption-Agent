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
        a, b, k = p
        return a*u + b*(x-a*u)*np.exp(-k*h) - y
    result = least_squares(residual, [1.985, 1.0, 1.0], bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x

def predict(row):
    a, b, k = _params
    return float(a*row['u_now'] + b*(row['x_now']-a*row['u_now'])*np.exp(-k*row['H']))