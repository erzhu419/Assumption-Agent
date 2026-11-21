import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    du = np.array([r['u_now'] - r['u_past'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        gain, rate, k_eq = p
        return gain * du * np.exp(-rate * h) + k_eq * u - y
    result = least_squares(residual, [-2.0, 1.0, 0.3], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]), xtol=1e-12, ftol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    gain, rate, k_eq = _params
    return float(gain * (row['u_now'] - row['u_past']) * np.exp(-rate * row['H']) + k_eq * row['u_now'])