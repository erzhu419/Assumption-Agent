import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    result = least_squares(residual, x0=[1.0, 1.0, 0.5, 10.0], bounds=([0.0, 0.0, 1e-4, 1e-3], [20.0, 20.0, 100.0, 1000.0]), max_nfev=20000)
    _PARAMS = result.x
    return tuple(float(v) for v in _PARAMS)

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))