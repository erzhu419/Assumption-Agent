import math
import numpy as np
from scipy.optimize import least_squares

_SCALE = 0.02
_params = None

def fit(train_rows):
    global _params
    data = np.array([[r['u_past'], r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float)
    past, now, horizon, target = data.T
    step = (past - now) / _SCALE

    def residual(p):
        gain, amplitude, log_tau = p
        return gain * now + amplitude * step * np.exp(-horizon / np.exp(log_tau)) - target

    _params = least_squares(residual, [25.0, 2.0, 0.0], bounds=([-np.inf, -np.inf, -12.0], [np.inf, np.inf, 12.0]), x_scale='jac').x

def predict(row):
    gain, amplitude, log_tau = _params
    step = (row['u_past'] - row['u_now']) / _SCALE
    return float(gain * row['u_now'] + amplitude * step * math.exp(-row['H'] / math.exp(log_tau)))