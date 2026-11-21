import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    u, s, h, y = np.array([[r[k] for k in ('u_now', 's_grad', 'H', 'y')] for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _evaluate(p, u, s, h) - y,
                           [8.0, 0.7, 0.35, 1.8, 12.0],
                           bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6],
                                   [np.inf, 1.0, np.inf, np.inf, np.inf]))
    _params = result.x

def predict(row):
    return float(_evaluate(_params, row['u_now'], row['s_grad'], row['H']))