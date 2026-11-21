import math
import numpy as np
from scipy.optimize import least_squares

SCALE = 0.02
_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / SCALE for r in train_rows])
    d = np.array([(r['u_now'] - r['u_past']) / SCALE for r in train_rows])
    h = np.array([r['H'] for r in train_rows])
    y = np.array([r['y'] for r in train_rows])
    def residual(p):
        gain, rate, k_eq = p
        return gain * d * np.exp(-rate * h) + k_eq * u - y
    _params = least_squares(residual, [-1.0, 1.0, 0.0], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate, k_eq = _params
    u = row['u_now'] / SCALE
    d = (row['u_now'] - row['u_past']) / SCALE
    return float(k_eq * u + gain * d * math.exp(-rate * row['H']))