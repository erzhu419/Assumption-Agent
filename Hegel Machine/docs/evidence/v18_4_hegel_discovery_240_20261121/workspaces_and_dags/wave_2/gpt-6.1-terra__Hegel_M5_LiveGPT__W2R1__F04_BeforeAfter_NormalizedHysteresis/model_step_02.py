import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    d = np.array([r['u_past'] - r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        a, k, b = p
        return x + d * (a * np.exp(-k * h) + b) - y
    result = least_squares(residual, [2.0, 1.0, -0.2], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row):
    a, k, b = _params
    return float(row['x_now'] + (row['u_past'] - row['u_now']) * (a * np.exp(-k * row['H']) + b))