import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    d = np.array([r['u_past'] - r['u_now'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return x + d * (p[0] * np.exp(-p[1] * h) + p[2]) - y
    result = least_squares(residual, [120.0, 1.5, -8.0], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]), max_nfev=10000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    _params = result.x

def predict(row):
    a, k, c = _params
    return float(row['x_now'] + (row['u_past'] - row['u_now']) * (a * np.exp(-k * row['H']) + c))