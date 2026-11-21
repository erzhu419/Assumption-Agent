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
    result = least_squares(residual, [2.0, np.log(2.0), 0.4], bounds=([-np.inf, -10.0, -np.inf], [np.inf, 10.0, np.inf]), max_nfev=10000)
    _params = result.x

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 5.0
    up = row['u_past'] / 5.0
    return float(gain * (un - up) * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * un)
