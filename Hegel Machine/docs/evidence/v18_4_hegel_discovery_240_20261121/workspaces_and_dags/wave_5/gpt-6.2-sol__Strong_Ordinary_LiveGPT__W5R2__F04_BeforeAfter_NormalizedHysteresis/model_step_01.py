import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None
_TAU_SCALE = 0.004

def fit(train_rows):
    global _PARAMS
    data = np.asarray([(r['u_past'], r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float)
    past, now, h, y = data.T
    now = now / _TAU_SCALE
    step = now - past / _TAU_SCALE

    def residual(p):
        gain, rate, eq = p
        return gain * step * np.exp(-rate * h) + eq * now - y

    _PARAMS = least_squares(residual, [1.0, 1.0, 0.5], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x
    return _PARAMS

def predict(row):
    gain, rate, eq = _PARAMS
    now = row['u_now'] / _TAU_SCALE
    past = row['u_past'] / _TAU_SCALE
    return float(gain * (now - past) * math.exp(-rate * row['H']) + eq * now)
