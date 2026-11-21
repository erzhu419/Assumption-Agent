import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    u, x, h, y = data.T

    def residual(p):
        gain, rate = p
        return gain * u * (-np.expm1(-rate * h)) + x * np.exp(-rate * h) - y

    _params = least_squares(residual, [float(u @ y / (u @ u)), 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate = _params
    z = -rate * row['H']
    return float(-gain * row['u_now'] * math.expm1(z) + row['x_now'] * math.exp(z))