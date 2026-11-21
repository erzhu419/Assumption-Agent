import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    d, u, h, y = np.asarray([(r['u_now'] - r['u_past'], r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        gain, tau, k_eq = p
        return gain * d * np.exp(-h / tau) + k_eq * u - y
    _params = least_squares(residual, [1.0, 1.0, 0.5], bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, tau, k_eq = _params
    u = row['u_now']
    return float(gain * (u - row['u_past']) * math.exp(-row['H'] / tau) + k_eq * u)
