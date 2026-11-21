import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    starts = [
        [1.0, 1.0, 1.0, 15.0],
        [1.0, 0.5, 0.5, 10.0],
        [2.0, 2.0, 1.0, 25.0],
        [0.5, 0.2, 0.5, 30.0],
    ]
    best = None
    for p0 in starts:
        res = least_squares(
            lambda p: np.asarray([_model(r, p) for r in train_rows]) - y,
            p0,
            bounds=([0.0, 1e-5, 0.0, 1e-5], [100.0, 1e4, 100.0, 1e4]),
            max_nfev=50000,
        )
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_model(row, _PARAMS))