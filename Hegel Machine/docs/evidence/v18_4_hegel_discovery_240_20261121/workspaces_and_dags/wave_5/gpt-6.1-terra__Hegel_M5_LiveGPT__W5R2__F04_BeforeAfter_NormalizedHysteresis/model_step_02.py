import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    du = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    residual = np.array([r['y'] - r['x_now'] for r in train_rows], dtype=float)
    def error(p):
        a, b, k = p
        return du * (a + b * np.exp(-k * h)) - residual
    result = least_squares(error, [150.0, 400.0, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf]), xtol=1e-13, ftol=1e-13, gtol=1e-13, max_nfev=5000)
    _params = result.x

def predict(row):
    a, b, k = _params
    return float(row['x_now'] + (row['u_now'] - row['u_past']) * (a + b * np.exp(-k * row['H'])))