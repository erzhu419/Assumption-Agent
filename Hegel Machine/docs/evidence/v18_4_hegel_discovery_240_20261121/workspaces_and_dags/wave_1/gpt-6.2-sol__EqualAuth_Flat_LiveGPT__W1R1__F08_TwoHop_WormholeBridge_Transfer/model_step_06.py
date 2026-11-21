import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root = np.sqrt(u)

    def residual(p):
        gain, half_sat, diverted, grad_scale, bypass = p
        load = grad_scale * s
        return gain * u / (half_sat + u) * (1.0 - diverted * load / (1.0 + load)) + bypass * root - y

    _params = least_squares(residual, [8.0, 1.0, 0.7, 1.0, 0.3], bounds=([0.0, 1e-8, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    gain, half_sat, diverted, grad_scale, bypass = _params
    u = float(row['u_now'])
    load = grad_scale * float(row['s_grad'])
    return float(gain * u / (half_sat + u) * (1.0 - diverted * load / (1.0 + load)) + bypass * math.sqrt(u))
