import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, step, x, y = np.asarray([(r['H'], r['u_past'] - r['u_now'], r['x_now'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        amplitude, offset, rate = p
        return x + step * (amplitude * np.exp(-rate * h) + offset) - y
    _params = least_squares(residual, [2.0, -0.3, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    amplitude, offset, rate = _params
    step = row['u_past'] - row['u_now']
    return float(row['x_now'] + step * (amplitude * math.exp(-rate * row['H']) + offset))