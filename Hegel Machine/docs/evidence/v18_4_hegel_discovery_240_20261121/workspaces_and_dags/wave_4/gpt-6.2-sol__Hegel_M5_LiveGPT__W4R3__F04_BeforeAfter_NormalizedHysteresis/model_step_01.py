import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x, d, h, y = np.asarray([(r['x_now'], r['u_now'] - r['u_past'], r['H'], r['y']) for r in train_rows], dtype=float).T
    def residual(p):
        gain, amplitude, tau = p
        return x + d * (gain - amplitude * np.exp(-h / tau)) - y
    _params = least_squares(residual, [100.0, 600.0, 0.8], bounds=([-np.inf, 0.0, 1e-6], [np.inf, np.inf, np.inf]), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    gain, amplitude, tau = _params
    d = row['u_now'] - row['u_past']
    return float(row['x_now'] + d * (gain - amplitude * math.exp(-row['H'] / tau)))