import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, a_slow, tau_fast, tau_slow = p
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return u * (a_fast * fast - a_slow * slow) - y

    _params = least_squares(residual, [2.0, 3.0, 0.5, 12.0], bounds=(1e-8, np.inf)).x
    return _params

def predict(row) -> float:
    a_fast, a_slow, tau_fast, tau_slow = _params
    h = row['H']
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return float(row['u_now'] * (a_fast * fast - a_slow * slow))