import math
import numpy as np
from scipy.optimize import least_squares

_SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    past, now, h, y = np.array([[r[k] for k in ('u_past', 'u_now', 'H', 'y')] for r in train_rows], dtype=float).T
    now = now / _SCALE
    delta = now - past / _SCALE

    def residual(p):
        gain, rate, equilibrium = p
        return equilibrium * now + gain * delta * np.exp(-rate * h) - y

    _params = least_squares(residual, [1.0, 1.0, 1.0], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    gain, rate, equilibrium = _params
    now = row['u_now'] / _SCALE
    delta = (row['u_now'] - row['u_past']) / _SCALE
    return float(equilibrium * now + gain * delta * math.exp(-rate * row['H']))
