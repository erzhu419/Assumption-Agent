import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    un = np.array([r['u_now'] / 0.02 for r in train_rows], dtype=float)
    up = np.array([r['u_past'] / 0.02 for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        gain, log_tau, k_eq = p
        return gain * (un - up) * np.exp(-h / np.exp(log_tau)) + k_eq * un - y
    results = [least_squares(residual, [2.0, np.log(tau), 0.45], bounds=([-1000, -10, -1000], [1000, 10, 1000])) for tau in (0.2, 0.5, 1.0, 2.0, 5.0)]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 0.02
    up = row['u_past'] / 0.02
    return float(gain * (un - up) * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * un)
