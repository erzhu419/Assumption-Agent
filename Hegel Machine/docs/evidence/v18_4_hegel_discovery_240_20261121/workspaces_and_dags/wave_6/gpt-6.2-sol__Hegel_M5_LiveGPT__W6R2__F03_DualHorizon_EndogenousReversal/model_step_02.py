import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.array([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(log_params):
        a_fast, a_slow, tau_fast, gap = np.exp(log_params)
        tau_slow = tau_fast + gap
        return u * (-a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)) - y

    result = least_squares(residual, np.log([1.5, 2.5, 0.5, 10.0]))
    a_fast, a_slow, tau_fast, gap = np.exp(result.x)
    _params = (a_fast, a_slow, tau_fast, tau_fast + gap)

def predict(row):
    a_fast, a_slow, tau_fast, tau_slow = _params
    h = row['H']
    return float(row['u_now'] * (-a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / tau_slow)))