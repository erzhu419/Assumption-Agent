import numpy as np
from scipy.optimize import least_squares

_keys = ('u_now', 's_grad', 'H')
_params = None

def _law(p, u, s, h):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return vmax * (1 - rho * s / (1 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    u, s, h = np.array([[r[k] for k in _keys] for r in train_rows], dtype=float).T
    y = np.array([r['y'] for r in train_rows], dtype=float)
    result = least_squares(lambda p: _law(p, u, s, h) - y,
                           [6.0, 0.7, 0.4, 2.0, 10.0],
                           bounds=([0.0, 0.0, 1e-6, 0.0, 1e-6],
                                   [np.inf, 1.0, np.inf, np.inf, np.inf]),
                           x_scale='jac')
    _params = result.x
    return _params

def predict(row):
    return float(_law(_params, *(row[k] for k in _keys)))