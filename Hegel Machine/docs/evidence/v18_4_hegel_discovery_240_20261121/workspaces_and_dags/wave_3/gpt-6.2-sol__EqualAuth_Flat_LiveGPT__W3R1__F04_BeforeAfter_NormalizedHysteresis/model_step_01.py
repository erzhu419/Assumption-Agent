import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.02
_params = None

def fit(train_rows):
    global _params
    data = np.asarray([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    past, now, horizon, y = data.T
    contrast = (now - past) / TAU_SCALE
    current = now / TAU_SCALE

    def residual(p):
        gain, tau, k_eq = p
        return gain * contrast * np.exp(-horizon / tau) + k_eq * current - y

    _params = least_squares(residual, [-1.0, max(float(np.median(horizon)), 0.01), 1.0], bounds=([-np.inf, 1e-6, -np.inf], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row):
    gain, tau, k_eq = _params
    now = row['u_now'] / TAU_SCALE
    past = row['u_past'] / TAU_SCALE
    return float(gain * (now - past) * math.exp(-row['H'] / tau) + k_eq * now)
