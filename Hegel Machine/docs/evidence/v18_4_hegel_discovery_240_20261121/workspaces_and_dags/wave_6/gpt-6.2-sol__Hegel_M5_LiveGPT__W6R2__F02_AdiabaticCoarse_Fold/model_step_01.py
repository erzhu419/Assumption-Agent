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
    gain0 = float(np.dot(u, y) / np.dot(u, u))
    _params = least_squares(residual, [gain0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]), gtol=1e-12).x

def predict(row):
    gain, rate = _params
    z = -rate * float(row['H'])
    return float(row['x_now'] * math.exp(z) - gain * row['u_now'] * math.expm1(z))
