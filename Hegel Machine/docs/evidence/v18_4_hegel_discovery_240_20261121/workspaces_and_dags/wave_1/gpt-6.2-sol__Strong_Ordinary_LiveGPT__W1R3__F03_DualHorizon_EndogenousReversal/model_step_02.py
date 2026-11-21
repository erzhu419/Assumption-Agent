import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([row['H'] for row in train_rows], dtype=float)
    u = np.array([row['u_now'] for row in train_rows], dtype=float)
    y = np.array([row['y'] for row in train_rows], dtype=float)

    def residual(p):
        amp_fast, amp_slow, tau_fast, gap = p
        tau_slow = tau_fast + gap
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return u * (amp_fast * fast - amp_slow * slow) - y

    result = least_squares(residual, [1.5, 2.3, 0.35, 12.0],
                           bounds=([0.0, 0.0, 1e-6, 1e-6], np.inf),
                           x_scale='jac')
    _params = result.x
    return _params

def predict(row) -> float:
    amp_fast, amp_slow, tau_fast, gap = _params
    h = row['H']
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / (tau_fast + gap))
    return float(row['u_now'] * (amp_fast * fast - amp_slow * slow))