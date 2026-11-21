import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) / TAU_SCALE for r in train_rows])
    step = u - np.array([float(r['u_past']) / TAU_SCALE for r in train_rows])
    h = np.array([float(r['H']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        gain, rate, k_eq = p
        return gain * step * np.exp(-rate * h) + k_eq * u - y
    _params = least_squares(residual, [1.0, 1.0, 0.0], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate, k_eq = _params
    u = float(row['u_now']) / TAU_SCALE
    step = (float(row['u_now']) - float(row['u_past'])) / TAU_SCALE
    return float(gain * step * math.exp(-rate * float(row['H'])) + k_eq * u)
