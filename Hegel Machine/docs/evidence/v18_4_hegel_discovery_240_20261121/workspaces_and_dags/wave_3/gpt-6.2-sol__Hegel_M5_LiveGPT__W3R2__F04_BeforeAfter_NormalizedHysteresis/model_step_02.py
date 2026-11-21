import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x, step, h, y = np.array([(r['x_now'], r['u_now'] - r['u_past'], r['H'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        steady, transient, rate = p
        return x + step * (steady + transient * np.exp(-rate * h)) - y
    _params = least_squares(residual, [0.5, 2.5, 1.5], bounds=([0.0, 0.0, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    steady, transient, rate = _params
    step = row['u_now'] - row['u_past']
    return float(row['x_now'] + step * (steady + transient * math.exp(-rate * row['H'])))
