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
        return k * u + (x - k * u) * np.exp(-a * h) - y
    result = least_squares(residual, [2.5, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    _params = result.x

def predict(row):
    k, a = _params
    u = float(row['u_now'])
    x = float(row['x_now'])
    h = float(row['H'])
    return float(k * u + (x - k * u) * np.exp(-a * h))
