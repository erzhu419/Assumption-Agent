import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, h):
    amp_fast, tau_fast, amp_slow, tau_gap = p
    tau_slow = tau_fast + tau_gap
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return u * (amp_fast * fast - amp_slow * slow)

def fit(train_rows):
    global _params
    u, h, y = np.array([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _response(p, u, h) - y,
                           x0=[1.6, 0.45, 2.5, 12.0],
                           bounds=(1e-8, np.inf), x_scale='jac')
    _params = result.x
    return _params

def predict(row):
    return float(_response(_params, float(row['u_now']), float(row['H'])))
