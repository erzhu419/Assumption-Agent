import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def fit(train_rows):
    global _params
    data = np.asarray([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    past, now, horizon, target = data.T
    now = now / TAU_SCALE
    step = now - past / TAU_SCALE

    def residual(params):
        gain, rate, k_eq = params
        return k_eq * now + gain * step * np.exp(-rate * horizon) - target

    _params = least_squares(residual, [1.0, 1.0, 0.5], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    gain, rate, k_eq = _params
    now = row['u_now'] / TAU_SCALE
    step = now - row['u_past'] / TAU_SCALE
    return float(k_eq * now + gain * step * math.exp(-rate * row['H']))