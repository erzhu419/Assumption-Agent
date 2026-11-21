import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    h = np.array([r['H'] for r in train_rows], dtype=float)
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)

    def residual(p):
        a_fast, a_slow, tau_fast, gap = p
        tau_slow = tau_fast + gap
        fast = -np.expm1(-h / tau_fast)
        slow = -np.expm1(-h / tau_slow)
        return u * (a_fast * fast - a_slow * slow) - y

    _params = least_squares(residual, [1.5, 2.5, 0.4, 12.0], bounds=([0, 0, 1e-6, 1e-6], np.inf), x_scale='jac').x

def predict(row) -> float:
    a_fast, a_slow, tau_fast, gap = _params
    h = row['H']
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / (tau_fast + gap))
    return float(row['u_now'] * (a_fast * fast - a_slow * slow))