import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(log_params, u, h):
    amp_fast, amp_slow, tau_fast, gap = np.exp(log_params)
    tau_slow = tau_fast + gap
    rise_fast = -np.expm1(-h / tau_fast)
    rise_slow = -np.expm1(-h / tau_slow)
    return u * (amp_fast * rise_fast - amp_slow * rise_slow)

def fit(train_rows):
    global _params
    u, h, y = np.asarray([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    start = np.log([1.7, 3.0, 0.45, 11.0])
    _params = least_squares(lambda p: _response(p, u, h) - y, start, ftol=1e-11, xtol=1e-11, gtol=1e-11).x

def predict(row):
    return float(_response(_params, float(row['u_now']), float(row['H'])))