import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [
        [0.9, 1.4, 0.3, 12.0],
        [1.2, 1.7, 0.5, 20.0],
        [2.0, 2.5, 1.0, 30.0],
        [0.6, 1.1, 0.2, 8.0],
    ]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([0.0, 0.0, 1e-4, 1e-3], [20.0, 20.0, 100.0, 500.0]), max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))