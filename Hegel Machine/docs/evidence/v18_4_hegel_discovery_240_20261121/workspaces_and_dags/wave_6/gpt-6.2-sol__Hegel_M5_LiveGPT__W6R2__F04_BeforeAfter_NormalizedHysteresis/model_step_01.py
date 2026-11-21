import math
import numpy as np
from scipy.optimize import least_squares

_scale = 0.004
_params = None

def fit(train_rows):
    global _params
    step, horizon, change = np.array([
        ((r['u_now'] - r['u_past']) / _scale, r['H'], r['y'] - r['x_now'])
        for r in train_rows
    ], dtype=float).T
    def residual(p):
        c, a, k = p
        return step * (c + a * np.exp(-k * horizon)) - change
    _params = least_squares(residual, [0.5, 2.0, 1.0], bounds=([-np.inf, -np.inf, 0.0], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row):
    c, a, k = _params
    step = (row['u_now'] - row['u_past']) / _scale
    return float(row['x_now'] + step * (c + a * math.exp(-k * row['H'])))
