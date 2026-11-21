import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        amp_fast, tau_fast, amp_slow, tau_gap = p
        tau_slow = tau_fast + tau_gap
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return u * (amp_fast * fast - amp_slow * slow) - y

    _params = least_squares(residual, [1.6, 0.45, 2.6, 11.5], bounds=([0, 1e-6, 0, 1e-6], np.inf)).x

def predict(row) -> float:
    amp_fast, tau_fast, amp_slow, tau_gap = _params
    h = row['H']
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / (tau_fast + tau_gap))
    return float(row['u_now'] * (amp_fast * fast - amp_slow * slow))