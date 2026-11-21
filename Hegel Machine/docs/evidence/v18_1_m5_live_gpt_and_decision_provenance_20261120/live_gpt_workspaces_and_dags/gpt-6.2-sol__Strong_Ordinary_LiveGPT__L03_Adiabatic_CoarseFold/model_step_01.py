import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, x, h, y = np.asarray([[r['u_now'], r['x_now'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        gain, rate = p
        weight = -np.expm1(-rate * h)
        return x + (gain * u - x) * weight - y

    _params = least_squares(residual, [2.0, 1.0], bounds=([0.0, 0.0], [np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    gain, rate = _params
    x = float(row['x_now'])
    weight = -math.expm1(-rate * float(row['H']))
    return float(x + (gain * float(row['u_now']) - x) * weight)
