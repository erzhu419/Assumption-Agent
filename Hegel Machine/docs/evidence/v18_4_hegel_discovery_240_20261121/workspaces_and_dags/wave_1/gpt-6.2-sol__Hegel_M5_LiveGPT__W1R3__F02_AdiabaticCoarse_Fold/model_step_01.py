import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, H, y = np.asarray([(r['u_now'], r['x_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    gain0 = float(np.dot(u, y) / np.dot(u, u))

    def residual(p):
        gain, rate = p
        return gain * u * (-np.expm1(-rate * H)) + x * np.exp(-rate * H) - y

    _params = least_squares(residual, [gain0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]), gtol=1e-12).x

def predict(row) -> float:
    gain, rate = _params
    z = -float(rate) * row['H']
    return float(gain * row['u_now'] * (-math.expm1(z)) + row['x_now'] * math.exp(z))