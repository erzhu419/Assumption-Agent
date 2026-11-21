import math
import numpy as np
from scipy.optimize import least_squares

_SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    now = np.array([r['u_now'] / _SCALE for r in rows], dtype=float)
    delta = now - np.array([r['u_past'] / _SCALE for r in rows], dtype=float)
    horizon = np.array([r['H'] for r in rows], dtype=float)
    target = np.array([r['y'] for r in rows], dtype=float)

    def residual(p):
        gain, rate, k_eq = p
        return gain * delta * np.exp(-rate * horizon) + k_eq * now - target

    result = least_squares(residual, [2.5, 1.25, 0.6], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return _params

def predict(row):
    gain, rate, k_eq = _params
    now = row['u_now'] / _SCALE
    past = row['u_past'] / _SCALE
    return float(gain * (now - past) * math.exp(-rate * row['H']) + k_eq * now)
