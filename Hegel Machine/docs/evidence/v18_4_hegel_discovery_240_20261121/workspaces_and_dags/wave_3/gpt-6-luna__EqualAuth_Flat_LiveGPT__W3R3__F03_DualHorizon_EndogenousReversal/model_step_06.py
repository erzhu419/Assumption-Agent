import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = [
        [1.5, 2.0, 0.8, 15.0],
        [1.0, 1.0, 0.6, 10.0],
        [2.0, 4.0, 1.2, 25.0],
        [1.0, 0.5, 0.5, 5.0]
    ]
    best = None
    for p0 in starts:
        result = least_squares(residual, p0, bounds=([0.0, 1e-5, 0.0, 1e-5], [20.0, 100.0, 20.0, 500.0]), max_nfev=20000)
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = best[1]
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))