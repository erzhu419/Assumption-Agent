import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'] / 5.0, (r['u_now'] - r['u_past']) / 5.0, r['H'], r['y']) for r in train_rows], dtype=float)
    u, step, h, y = data.T

    def residual(p):
        gain, rate, k_eq = p
        return gain * step * np.exp(-rate * h) + k_eq * u - y

    _params = least_squares(residual, [1.0, 1.0, 0.5], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    gain, rate, k_eq = _params
    u = row['u_now'] / 5.0
    step = (row['u_now'] - row['u_past']) / 5.0
    return float(k_eq * u + gain * step * math.exp(-rate * row['H']))
