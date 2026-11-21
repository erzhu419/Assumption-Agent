import math
import numpy as np
from scipy.optimize import least_squares

_p = None

def _response(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    gain = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return gain * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _p
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _response(p, u, s, h) - y,
                           [6.5, 0.7, 0.35, 1.4, 10.0],
                           bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6],
                                   [np.inf, 1.0, np.inf, np.inf, np.inf]),
                           x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _p = result.x
    return _p

def predict(row):
    vmax, rho, tau_fast, amp_adapt, tau_slow = _p
    u, s, h = row['u_now'], row['s_grad'], row['H']
    gain = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return float(-gain * math.expm1(-h / tau_fast) + amp_adapt * u * math.expm1(-h / tau_slow))