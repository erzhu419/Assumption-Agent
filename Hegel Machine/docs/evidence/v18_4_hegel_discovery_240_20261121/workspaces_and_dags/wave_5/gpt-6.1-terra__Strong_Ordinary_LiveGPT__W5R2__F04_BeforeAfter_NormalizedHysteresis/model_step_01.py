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
        gain, log_tau, k_eq = p
        return gain * (un - up) * np.exp(-h / np.exp(log_tau)) + k_eq * un - y
    candidates = [least_squares(residual, [g, np.log(t), 0.6], bounds=([-1000, -10, -1000], [1000, 10, 1000]), max_nfev=5000) for t in (0.1, 0.5, 2.0) for g in (1.0, 3.0)]
    _params = min(candidates, key=lambda r: np.sum(r.fun ** 2)).x

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 0.004
    up = row['u_past'] / 0.004
    return float(gain * (un - up) * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * un)
