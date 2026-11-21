import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    up = np.array([r['u_past'] / 5.0 for r in train_rows], dtype=float)
    un = np.array([r['u_now'] / 5.0 for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        gain, log_tau, k_eq = p
        return gain * (un - up) * np.exp(-h / np.exp(log_tau)) + k_eq * un - y
    candidates = [least_squares(residual, [1.0, np.log(t), 0.5], bounds=([-100.0, -10.0, -100.0], [100.0, 10.0, 100.0])) for t in (0.2, 1.0, 5.0, 20.0)]
    _params = min(candidates, key=lambda result: np.sum(result.fun ** 2)).x

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 5.0
    up = row['u_past'] / 5.0
    return float(gain * (un - up) * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * un)
