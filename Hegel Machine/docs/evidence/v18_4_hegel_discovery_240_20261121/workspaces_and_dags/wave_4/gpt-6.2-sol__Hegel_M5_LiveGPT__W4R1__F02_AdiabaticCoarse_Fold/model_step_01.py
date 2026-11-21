import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        gain, rate = p
        return x * np.exp(-rate * h) - gain * u * np.expm1(-rate * h) - y

    _params = least_squares(residual, [2.0, 1.0], bounds=([0.0, 0.0], [np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    gain, rate = _params
    z = -rate * row['H']
    return float(row['x_now'] * math.exp(z) - gain * row['u_now'] * math.expm1(z))
