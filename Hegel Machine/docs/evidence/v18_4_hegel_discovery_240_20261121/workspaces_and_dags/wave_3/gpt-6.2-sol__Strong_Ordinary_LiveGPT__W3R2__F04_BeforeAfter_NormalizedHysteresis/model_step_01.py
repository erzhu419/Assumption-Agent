import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def fit(train_rows):
    global _params
    past, now, h, y = np.array([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    u = now / TAU_SCALE
    delta = (now - past) / TAU_SCALE
    def residual(p):
        gain, rate, k_eq = p
        return gain * delta * np.exp(-rate * h) + k_eq * u - y
    _params = least_squares(residual, [2.0, 1.0, 0.5], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate, k_eq = _params
    u = row['u_now'] / TAU_SCALE
    delta = (row['u_now'] - row['u_past']) / TAU_SCALE
    return float(gain * delta * math.exp(-rate * row['H']) + k_eq * u)
