import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u_past = np.array([r['u_past'] for r in train_rows], dtype=float)
    u_now = np.array([r['u_now'] for r in train_rows], dtype=float)
    H = np.array([r['H'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        gain, tau, k_eq = p
        return gain * (u_now - u_past) * np.exp(-H / tau) + k_eq * u_now - y

    result = least_squares(residual, x0=np.array([1.0, 1.0, 0.5]), bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    gain, tau, k_eq = _params
    return float(gain * (row['u_now'] - row['u_past']) * math.exp(-row['H'] / tau) + k_eq * row['u_now'])