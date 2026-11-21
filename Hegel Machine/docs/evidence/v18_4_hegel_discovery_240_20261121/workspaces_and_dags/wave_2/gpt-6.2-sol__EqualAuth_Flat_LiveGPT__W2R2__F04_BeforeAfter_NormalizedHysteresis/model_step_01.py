import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([[r[k] for k in ('u_past', 'u_now', 'H', 'y')] for r in train_rows], dtype=float)
    past, now, horizon, y = data.T
    now = now / 5.0
    delta = now - past / 5.0

    def residual(p):
        gain, tau, k_eq = p
        return gain * delta * np.exp(-horizon / tau) + k_eq * now - y

    _params = least_squares(residual, (1.0, 1.0, 0.5), bounds=([-np.inf, 1e-8, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    gain, tau, k_eq = _params
    now = row['u_now'] / 5.0
    delta = now - row['u_past'] / 5.0
    return float(gain * delta * math.exp(-row['H'] / tau) + k_eq * now)
