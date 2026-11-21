import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    p = np.array([r['u_past'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(z):
        gain, log_tau, k_eq = z
        return gain * (u - p) * np.exp(-h / np.exp(log_tau)) + k_eq * u - y
    result = least_squares(residual, [1.7, 0.0, 0.645])
    _params = result.x

def predict(row):
    gain, log_tau, k_eq = _params
    return float(gain * (row['u_now'] - row['u_past']) * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * row['u_now'])