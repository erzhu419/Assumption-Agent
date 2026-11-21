import math
import numpy as np
from scipy.optimize import least_squares

_KEYS = ('u_now', 's_grad', 'H')
_params = None

def _response(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_gap = p
    tau_slow = tau_fast + tau_gap
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base * (-np.expm1(-h / tau_fast)) - amp_adapt * u * (-np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u, s, h = np.asarray([[r[k] for k in _KEYS] for r in rows], dtype=float).T
    y = np.asarray([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _response(p, u, s, h) - y,
                           [5.0, 0.7, 0.2, 1.5, 12.0],
                           bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6],
                                   [math.inf, 1.0, math.inf, math.inf, math.inf]),
                           x_scale='jac')
    _params = result.x

def predict(row):
    return float(_response(_params, *(float(row[k]) for k in _KEYS)))
