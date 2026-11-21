import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T
    k0 = float(np.dot(u, y) / np.dot(u, u))

    def residual(p):
        k, a = p
        decay = np.exp(-a * h)
        return k * u * (-np.expm1(-a * h)) + x * decay - y

    result = least_squares(residual, [k0, 1.0], bounds=([-np.inf, 0.0], [np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = tuple(map(float, result.x))

def predict(row):
    k, a = _params
    z = -a * row['H']
    return float(k * row['u_now'] * (-math.expm1(z)) + row['x_now'] * math.exp(z))
