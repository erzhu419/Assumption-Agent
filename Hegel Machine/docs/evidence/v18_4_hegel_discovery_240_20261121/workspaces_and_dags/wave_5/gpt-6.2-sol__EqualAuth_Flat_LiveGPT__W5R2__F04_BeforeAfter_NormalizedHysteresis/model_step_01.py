import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float) / 0.004
    step = u - np.array([r['u_past'] for r in rows], dtype=float) / 0.004
    h = np.array([r['H'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        gain, tau, k_eq = p
        return gain * step * np.exp(-h / tau) + k_eq * u - y

    _params = least_squares(residual, [1.0, float(np.median(h)), 0.5], bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row):
    gain, tau, k_eq = _params
    u = row['u_now'] / 0.004
    step = (row['u_now'] - row['u_past']) / 0.004
    return float(gain * step * math.exp(-row['H'] / tau) + k_eq * u)
