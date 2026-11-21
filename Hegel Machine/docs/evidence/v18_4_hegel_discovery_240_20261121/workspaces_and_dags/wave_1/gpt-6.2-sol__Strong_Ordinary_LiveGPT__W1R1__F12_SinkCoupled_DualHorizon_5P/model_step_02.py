import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(u, s, h, p):
    vmax, rho, log_tau_fast, amp_slow, log_tau_slow = p
    fast = -np.expm1(-h / math.exp(log_tau_fast))
    slow = -np.expm1(-h / math.exp(log_tau_slow))
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base * fast - amp_slow * u * slow

def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r['u_now'], r['s_grad'], r['H'], r['y']] for r in train_rows], dtype=float).T
    start = [6.0, 0.7, math.log(0.5), 2.0, math.log(10.0)]
    lower = [0.0, 0.0, -20.0, 0.0, -20.0]
    upper = [np.inf, 1.0, 20.0, np.inf, 20.0]
    _params = least_squares(lambda p: _response(u, s, h, p) - y, start, bounds=(lower, upper), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    return float(_response(float(row['u_now']), float(row['s_grad']), float(row['H']), _params))