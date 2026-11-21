import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.array([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        gain, rate = p
        z = -rate * h
        return x * np.exp(z) - gain * u * np.expm1(z) - y

    _params = least_squares(residual, [1.0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate = _params
    z = -rate * row['H']
    return float(row['x_now'] * math.exp(z) - gain * row['u_now'] * math.expm1(z))
