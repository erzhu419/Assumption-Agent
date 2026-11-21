import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, h):
    amp_fast, amp_slow, log_fast, log_gap = p
    tau_fast = np.exp(log_fast)
    tau_slow = tau_fast + np.exp(log_gap)
    fast = -np.expm1(-h / tau_fast)
    slow = -np.expm1(-h / tau_slow)
    return u * (amp_fast * fast - amp_slow * slow)

def fit(train_rows):
    global _params
    u, h, y = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(
        lambda p: _response(p, u, h) - y,
        [1.5, 2.7, math.log(0.4), math.log(11.0)],
        bounds=([0.0, 0.0, -12.0, -12.0], [np.inf, np.inf, 12.0, 12.0]),
    )
    _params = result.x
    return _params

def predict(row):
    return float(_response(_params, float(row['u_now']), float(row['H'])))
