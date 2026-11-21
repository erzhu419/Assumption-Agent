import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    un = np.array([r['u_now'] / 5.0 for r in train_rows], dtype=float)
    up = np.array([r['u_past'] / 5.0 for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * (un - up) * np.exp(-h / np.exp(p[1])) + p[2] * un - y
    best = None
    for tau in (0.2, 1.0, 5.0, 20.0):
        result = least_squares(residual, [1.0, np.log(tau), 0.3], bounds=([-np.inf, -8, -np.inf], [np.inf, 8, np.inf]))
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 5.0
    up = row['u_past'] / 5.0
    return float(gain * (un - up) * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * un)
