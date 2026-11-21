import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    d = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        a, b, c = p
        return x + d * (a + b * np.exp(-c * h)) - y
    result = least_squares(residual, [20.0, 100.0, 0.8], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x

def predict(row):
    a, b, c = _params
    d = float(row['u_now']) - float(row['u_past'])
    return float(row['x_now'] + d * (a + b * np.exp(-c * float(row['H']))))
