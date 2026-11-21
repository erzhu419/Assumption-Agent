import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    up = np.array([r['u_past'] for r in train_rows], dtype=float)
    un = np.array([r['u_now'] for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        gain, log_tau, k_eq = p
        return gain * (un - up) * np.exp(-h / np.exp(log_tau)) + k_eq * un - y
    fits = [least_squares(residual, [g, t, k], bounds=([-100, -8, -100], [100, 8, 100])) for g, t, k in [(3, 0, 0.6), (1, -1, 0), (5, 1, 1)]]
    _params = min(fits, key=lambda result: np.sum(result.fun ** 2)).x
    return _params

def predict(row):
    gain, log_tau, k_eq = _params
    return float(gain * (row['u_now'] - row['u_past']) * math.exp(-row['H'] / math.exp(log_tau)) + k_eq * row['u_now'])