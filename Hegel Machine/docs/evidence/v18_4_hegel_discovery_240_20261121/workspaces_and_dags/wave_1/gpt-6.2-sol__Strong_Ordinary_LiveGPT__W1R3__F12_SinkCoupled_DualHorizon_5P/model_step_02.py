import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s, h):
    vmax, rho, tau_fast, amp_slow, tau_slow = p
    base = vmax * (u / (1.2 + u)) * (1.0 - rho * s / (1.0 + s))
    return base * (-np.expm1(-h / tau_fast)) - amp_slow * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    u, s, h, y = np.asarray([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _response(p, u, s, h) - y,
                           x0=[5.0, 0.7, 0.3, 2.0, 10.0],
                           bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6],
                                   [math.inf, 1.0, math.inf, math.inf, math.inf]),
                           x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    return float(_response(_params, row['u_now'], row['s_grad'], row['H']))