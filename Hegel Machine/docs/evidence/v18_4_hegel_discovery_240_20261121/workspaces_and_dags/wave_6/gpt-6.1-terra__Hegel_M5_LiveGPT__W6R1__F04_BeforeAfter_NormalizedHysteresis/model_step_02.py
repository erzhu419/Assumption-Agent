import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    d = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    x = np.array([r['x_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    # A saturating decay accommodates a changing apparent relaxation rate.
    def residual(p):
        a, b, c = p
        return x + d * a * np.exp(-c * np.log1p(b * h)) - y
    result = least_squares(residual, [0.4, 0.3, 2.2], bounds=([0.0, 1e-8, 0.0], [10.0, 100.0, 100.0]), max_nfev=20000, xtol=1e-13, ftol=1e-13, gtol=1e-13)
    _params = result.x

def predict(row):
    a, b, c = _params
    return float(row['x_now'] + (row['u_now'] - row['u_past']) * a * np.exp(-c * np.log1p(b * row['H'])))
