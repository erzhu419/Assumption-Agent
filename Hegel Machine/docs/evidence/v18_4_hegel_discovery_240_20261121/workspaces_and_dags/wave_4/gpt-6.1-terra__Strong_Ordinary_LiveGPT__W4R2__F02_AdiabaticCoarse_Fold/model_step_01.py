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
        eq = p[0] * u
        return eq + (x - eq) * np.exp(-p[1] * h) - y
    result = least_squares(residual, [2.2, 1.5], bounds=([-np.inf, 0.0], [np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row):
    k, a = _params
    eq = k * float(row['u_now'])
    return float(eq + (float(row['x_now']) - eq) * np.exp(-a * float(row['H'])))