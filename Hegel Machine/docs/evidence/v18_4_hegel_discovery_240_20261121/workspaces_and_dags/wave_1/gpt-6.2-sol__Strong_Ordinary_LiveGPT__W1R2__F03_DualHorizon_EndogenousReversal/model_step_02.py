import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(log_params):
        a_fast, a_slow, tau_fast, tau_gap = np.exp(log_params)
        tau_slow = tau_fast + tau_gap
        gain = -a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)
        return u * gain - y

    result = least_squares(residual, np.log([1.5, 2.4, 0.35, 13.0]))
    a_fast, a_slow, tau_fast, tau_gap = np.exp(result.x)
    _params = (a_fast, a_slow, tau_fast, tau_fast + tau_gap)

def predict(row):
    a_fast, a_slow, tau_fast, tau_slow = _params
    h = row['H']
    gain = -a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / tau_slow)
    return float(row['u_now'] * gain)