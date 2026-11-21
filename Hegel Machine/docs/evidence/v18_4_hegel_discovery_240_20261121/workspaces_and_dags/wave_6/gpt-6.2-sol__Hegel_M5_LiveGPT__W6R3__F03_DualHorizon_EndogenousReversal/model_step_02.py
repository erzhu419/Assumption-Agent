import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, h, y = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T

    def residual(log_params):
        a_fast, a_slow, tau_fast, gap = np.exp(log_params)
        tau_slow = tau_fast + gap
        return u * (-a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)) - y

    result = least_squares(residual, np.log([1.0, 2.0, 0.5, 10.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    a_fast, a_slow, tau_fast, gap = np.exp(result.x)
    _params = (a_fast, a_slow, tau_fast, tau_fast + gap)
    return _params

def predict(row) -> float:
    a_fast, a_slow, tau_fast, tau_slow = _params
    h = row['H']
    return float(row['u_now'] * (-a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / tau_slow)))