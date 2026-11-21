import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    un = np.array([r['u_now'] / 0.004 for r in train_rows], dtype=float)
    up = np.array([r['u_past'] / 0.004 for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return p[0] * (un - up) * np.exp(-h / np.exp(p[1])) + p[2] * un - y
    fits = [least_squares(residual, [-1.0, np.log(t), 0.3], bounds=([-np.inf, -10, -np.inf], [np.inf, 10, np.inf])) for t in (0.1, 0.5, 2.0)]
    _params = min(fits, key=lambda f: np.sum(f.fun ** 2)).x

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 0.004
    up = row['u_past'] / 0.004
    return float(gain * (un - up) * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * un)
