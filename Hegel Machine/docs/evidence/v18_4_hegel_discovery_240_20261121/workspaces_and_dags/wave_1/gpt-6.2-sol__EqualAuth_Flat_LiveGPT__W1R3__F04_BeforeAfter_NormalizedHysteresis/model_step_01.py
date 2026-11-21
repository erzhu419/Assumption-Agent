import math
import numpy as np
from scipy.optimize import least_squares

_scale = 0.004
_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_past'], r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float)
    past, now, horizon, y = data.T
    step = (now - past) / _scale
    level = now / _scale
    def residual(p):
        gain, rate, eq = p
        return gain * step * np.exp(-rate * horizon) + eq * level - y
    _params = least_squares(residual, [1.0, 1.0, 0.0], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate, eq = _params
    return float(gain * (row['u_now'] - row['u_past']) / _scale * math.exp(-rate * row['H']) + eq * row['u_now'] / _scale)
