import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x, step, h, y = np.asarray([(r['x_now'], r['u_now'] - r['u_past'], r['H'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        c, a, k = p
        return x + step * (c + a * np.exp(-k * h)) - y
    _params = least_squares(residual, [0.1, 0.3, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    c, a, k = _params
    step = row['u_now'] - row['u_past']
    return float(row['x_now'] + step * (c + a * math.exp(-k * row['H'])))
