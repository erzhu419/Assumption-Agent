import math
import numpy as np
from scipy.optimize import least_squares

_keys = ('u_past', 'u_now', 'x_now', 'H')
_params = None

def fit(train_rows):
    global _params
    data = np.array([[r[k] for k in _keys + ('y',)] for r in train_rows], dtype=float)
    past, now, x, h, y = data.T
    du = now - past

    def residual(p):
        c, a, log_tau = p
        return c * now + (x - c * past + a * du) * np.exp(-h / np.exp(log_tau)) - y

    _params = least_squares(residual, [0.5, 1.0, 0.0], ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row):
    c, a, log_tau = _params
    past, now, x, h = (float(row[k]) for k in _keys)
    return float(c * now + (x - c * past + a * (now - past)) * math.exp(-h / math.exp(log_tau)))
