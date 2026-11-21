import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    d = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        a, b, k = p
        return x + d * (a + b * np.exp(-k * h)) - y
    _params = least_squares(residual, [0.5, 1.5, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row):
    a, b, k = _params
    return float(row['x_now'] + (row['u_now'] - row['u_past']) * (a + b * np.exp(-k * row['H'])))
