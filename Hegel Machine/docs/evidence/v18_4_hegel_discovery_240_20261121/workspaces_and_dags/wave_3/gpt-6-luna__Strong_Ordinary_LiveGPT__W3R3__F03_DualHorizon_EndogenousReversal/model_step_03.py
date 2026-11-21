import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    u = float(row['u_now'])
    H = float(row['H'])
    return u * (amp_fast * (1.0 - math.exp(-H / tau_fast)) - amp_slow * (1.0 - math.exp(-H / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = ([1.0, 1.0, 0.3, 10.0], [2.0, 1.0, 0.5, 20.0], [1.0, 2.0, 0.2, 5.0])
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=([0.0, 0.0, 1e-5, 1e-5], [100.0, 100.0, 1e4, 1e4]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))