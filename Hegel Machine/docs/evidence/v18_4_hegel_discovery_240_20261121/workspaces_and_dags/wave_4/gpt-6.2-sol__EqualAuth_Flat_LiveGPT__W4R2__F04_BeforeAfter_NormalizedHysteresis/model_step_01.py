import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.004
_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'] / TAU_SCALE, (r['u_now'] - r['u_past']) / TAU_SCALE, r['H'], r['y']) for r in train_rows], dtype=float)
    now, step, horizon, y = data.T

    def residual(p):
        gain, rate, k_eq = p
        return k_eq * now + gain * step * np.exp(-rate * horizon) - y

    _params = least_squares(residual, [-1.0, 1.0, 0.3], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate, k_eq = _params
    now = row['u_now'] / TAU_SCALE
    step = (row['u_now'] - row['u_past']) / TAU_SCALE
    return float(k_eq * now + gain * step * math.exp(-rate * row['H']))
