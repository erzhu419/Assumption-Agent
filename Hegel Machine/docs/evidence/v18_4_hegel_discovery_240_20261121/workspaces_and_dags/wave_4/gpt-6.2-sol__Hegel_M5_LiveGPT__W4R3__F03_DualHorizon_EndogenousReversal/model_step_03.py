import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, h):
    amp_fast, tau_fast, amp_slow, gap = np.exp(p)
    tau_slow = tau_fast + gap
    return u * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    u, h, y = np.array([(r['u_now'], r['H'], r['y']) for r in train_rows], dtype=float).T
    start = np.log([1.5, 0.4, 2.5, 12.0])
    _params = least_squares(lambda p: _response(p, u, h) - y, start, bounds=(-12.0, 12.0)).x

def predict(row) -> float:
    return float(_response(_params, row['u_now'], row['H']))
