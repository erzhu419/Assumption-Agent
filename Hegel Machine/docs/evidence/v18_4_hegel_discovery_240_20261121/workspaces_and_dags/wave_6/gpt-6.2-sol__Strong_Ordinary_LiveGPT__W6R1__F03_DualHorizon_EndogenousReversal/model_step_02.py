import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(h, u, p):
    amp_fast, tau_fast, amp_slow, tau_gap = np.exp(p)
    tau_slow = tau_fast + tau_gap
    return u * (-amp_fast * np.expm1(-h / tau_fast) + amp_slow * np.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _response(h, u, p) - y,
                           np.log([1.4, 0.4, 2.4, 12.0]))
    _params = result.x
    return _params

def predict(row) -> float:
    amp_fast, tau_fast, amp_slow, tau_gap = np.exp(_params)
    h = float(row['H'])
    return float(row['u_now'] * (-amp_fast * math.expm1(-h / tau_fast) + amp_slow * math.expm1(-h / (tau_fast + tau_gap))))
