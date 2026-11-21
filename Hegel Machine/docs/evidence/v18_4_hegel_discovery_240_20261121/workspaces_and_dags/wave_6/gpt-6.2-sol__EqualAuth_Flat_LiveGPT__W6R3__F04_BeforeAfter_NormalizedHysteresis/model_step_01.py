import math
import numpy as np
from scipy.optimize import least_squares

_SCALE = 0.02
_params = None


def fit(train_rows):
    global _params
    data = np.array([(r['u_now'] / _SCALE, r['u_past'] / _SCALE, r['H'], r['y']) for r in train_rows], dtype=float)
    u_now, u_past, h, y = data.T
    contrast = u_now - u_past

    def residual(params):
        gain, log_tau, k_eq = params
        return gain * contrast * np.exp(-h / math.exp(log_tau)) + k_eq * u_now - y

    _params = least_squares(residual, [1.0, math.log(float(np.median(h))), 1.0], bounds=([-np.inf, -20.0, -np.inf], [np.inf, 20.0, np.inf])).x


def predict(row) -> float:
    gain, log_tau, k_eq = _params
    contrast = (row['u_now'] - row['u_past']) / _SCALE
    return float(gain * contrast * math.exp(-row['H'] / math.exp(log_tau)) + k_eq * row['u_now'] / _SCALE)
