import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_tau_scale = 1.0

def fit(train_rows):
    global _params
    data = np.asarray([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    past, now, horizon, target = data.T
    past = past / _tau_scale
    now = now / _tau_scale
    def residual(p):
        gain, rate, k_eq = p
        return gain * (now - past) * np.exp(-rate * horizon) + k_eq * now - target
    _params = least_squares(residual, [1.0, 1.0, 1.0], bounds=([-np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    gain, rate, k_eq = _params
    now = row['u_now'] / _tau_scale
    past = row['u_past'] / _tau_scale
    return float(gain * (now - past) * math.exp(-rate * row['H']) + k_eq * now)
