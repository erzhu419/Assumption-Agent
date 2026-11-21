import math
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
        tau = np.exp(log_tau)
        return gain * (un - up) * np.exp(-h / tau) + k_eq * un - y
    fits = [least_squares(residual, [2.0, math.log(t), 0.35], bounds=([-np.inf, -12.0, -np.inf], [np.inf, 12.0, np.inf])) for t in (0.1, 0.5, 1.0, 3.0)]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    gain, log_tau, k_eq = _params
    un = row['u_now'] / 0.004
    up = row['u_past'] / 0.004
    return float(gain * (un - up) * math.exp(-row['H'] / math.exp(log_tau)) + k_eq * un)
