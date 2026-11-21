import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 0.02 for r in train_rows], dtype=float)
    d = np.array([(r['u_now'] - r['u_past']) / 0.02 for r in train_rows], dtype=float)
    h = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        gain, log_tau, k_eq = p
        return gain * d * np.exp(-h / np.exp(log_tau)) + k_eq * u - y
    fits = [least_squares(residual, [2.0, np.log(t), 0.45], bounds=([-np.inf, -10, -np.inf], [np.inf, 10, np.inf])) for t in (0.1, 0.5, 1.0, 3.0)]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    gain, log_tau, k_eq = _params
    return float(gain * ((row['u_now'] - row['u_past']) / 0.02) * np.exp(-row['H'] / np.exp(log_tau)) + k_eq * row['u_now'] / 0.02)
