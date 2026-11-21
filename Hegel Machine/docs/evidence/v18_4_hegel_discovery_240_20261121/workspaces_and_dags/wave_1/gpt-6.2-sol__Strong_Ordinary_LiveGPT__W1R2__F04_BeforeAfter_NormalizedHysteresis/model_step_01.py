import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def fit(train_rows):
    global _params
    a = np.asarray([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    past, now, h, y = a.T
    u = now / TAU_SCALE
    step = u - past / TAU_SCALE
    def residual(p):
        gain, rate, k_eq = p
        return gain * step * np.exp(-rate * h) + k_eq * u - y
    _params = least_squares(residual, [1.0, 1.0, 0.5], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    gain, rate, k_eq = _params
    u = row['u_now'] / TAU_SCALE
    step = (row['u_now'] - row['u_past']) / TAU_SCALE
    return float(gain * step * math.exp(-rate * row['H']) + k_eq * u)
