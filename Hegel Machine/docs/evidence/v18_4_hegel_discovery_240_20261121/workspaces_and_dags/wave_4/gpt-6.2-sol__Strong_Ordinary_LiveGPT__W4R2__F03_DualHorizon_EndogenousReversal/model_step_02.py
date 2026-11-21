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
        gain = -a_fast * np.expm1(-h / tau_fast) + a_slow * np.expm1(-h / tau_slow)
        return u * gain - y

    start = np.log([1.5, 2.0, 0.5, 10.0])
    result = least_squares(residual, start, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    a_fast, a_slow, tau_fast, gap = np.exp(result.x)
    _params = (a_fast, a_slow, tau_fast, tau_fast + gap)

def predict(row) -> float:
    a_fast, a_slow, tau_fast, tau_slow = _params
    h = row['H']
    gain = -a_fast * math.expm1(-h / tau_fast) + a_slow * math.expm1(-h / tau_slow)
    return float(row['u_now'] * gain)
