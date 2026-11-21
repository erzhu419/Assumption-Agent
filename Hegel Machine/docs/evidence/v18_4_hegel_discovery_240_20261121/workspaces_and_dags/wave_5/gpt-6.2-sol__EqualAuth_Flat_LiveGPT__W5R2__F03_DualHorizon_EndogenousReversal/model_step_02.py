import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, h, y = np.asarray([[r['u_now'], r['H'], r['y']] for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, a_slow, tau_fast, gap = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / (tau_fast + gap))
        return u * (a_fast * fast - a_slow * slow) - y

    result = least_squares(residual, [1.0, 2.0, 0.5, 10.0], bounds=([0.0, 0.0, 1e-8, 1e-8], np.inf), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row) -> float:
    a_fast, a_slow, tau_fast, gap = _params
    h = float(row['H'])
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / (tau_fast + gap))
    return float(row['u_now'] * (a_fast * fast - a_slow * slow))