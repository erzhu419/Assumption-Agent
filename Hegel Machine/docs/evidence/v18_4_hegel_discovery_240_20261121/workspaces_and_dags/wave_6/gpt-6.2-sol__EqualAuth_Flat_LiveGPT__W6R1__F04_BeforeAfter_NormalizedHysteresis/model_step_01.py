import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_scale = 5.0

def fit(train_rows):
    global _params
    rows = list(train_rows)
    past, now, horizon, target = np.array([[r['u_past'], r['u_now'], r['H'], r['y']] for r in rows], dtype=float).T
    step = (now - past) / _scale
    level = now / _scale

    def residual(p):
        gain, tau, k_eq = p
        return gain * step * np.exp(-horizon / tau) + k_eq * level - target

    _params = least_squares(residual, [1.0, 1.0, 0.0], bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row):
    gain, tau, k_eq = _params
    step = (row['u_now'] - row['u_past']) / _scale
    level = row['u_now'] / _scale
    return float(gain * step * math.exp(-row['H'] / tau) + k_eq * level)
