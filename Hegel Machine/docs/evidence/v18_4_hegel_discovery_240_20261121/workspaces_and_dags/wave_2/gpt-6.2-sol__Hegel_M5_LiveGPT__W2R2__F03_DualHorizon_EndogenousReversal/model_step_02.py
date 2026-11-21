import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h, u, y = np.asarray([(r['H'], r['u_now'], r['y']) for r in train_rows], dtype=float).T

    def residual(log_params):
        fast_amp, slow_amp, fast_tau, tau_gap = np.exp(log_params)
        slow_tau = fast_tau + tau_gap
        fast = -np.expm1(-h / fast_tau)
        slow = -np.expm1(-h / slow_tau)
        return u * (fast_amp * fast - slow_amp * slow) - y

    result = least_squares(residual, np.log([1.46, 2.67, 0.32, 13.03]))
    fast_amp, slow_amp, fast_tau, tau_gap = np.exp(result.x)
    _params = tuple(map(float, (fast_amp, slow_amp, fast_tau, fast_tau + tau_gap)))

def predict(row) -> float:
    fast_amp, slow_amp, fast_tau, slow_tau = _params
    h = float(row['H'])
    fast = -math.expm1(-h / fast_tau)
    slow = -math.expm1(-h / slow_tau)
    return float(row['u_now'] * (fast_amp * fast - slow_amp * slow))
