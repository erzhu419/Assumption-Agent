import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(p):
        a_fast, a_slow, log_fast, log_gap = p
        tau_fast = math.exp(log_fast)
        tau_slow = tau_fast + math.exp(log_gap)
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return u * (a_fast * fast - a_slow * slow) - y

    result = least_squares(residual, [1.3, 2.7, math.log(0.3), math.log(13.0)], bounds=([0.0, 0.0, -12.0, -12.0], [np.inf, np.inf, 12.0, 12.0]))
    a_fast, a_slow, log_fast, log_gap = result.x
    tau_fast = math.exp(log_fast)
    _params = (a_fast, a_slow, tau_fast, tau_fast + math.exp(log_gap))

def predict(row):
    a_fast, a_slow, tau_fast, tau_slow = _params
    h = row['H']
    return float(row['u_now'] * (-a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / tau_slow)))