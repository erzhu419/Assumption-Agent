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
        k, b, a = p
        steady = k * u + b
        return steady + (x - steady) * np.exp(-a * h) - y
    result = least_squares(residual, [2.5, 0.0, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, 20.0]))
    _params = result.x

def predict(row):
    k, b, a = _params
    steady = k * float(row['u_now']) + b
    return float(steady + (float(row['x_now']) - steady) * np.exp(-a * float(row['H'])))