import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    af, a_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (af * (1.0 - math.exp(-h / tau_fast)) - a_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    starts = [
        [1.2, 0.8, 0.5, 12.0],
        [1.0, 0.7, 1.0, 20.0],
        [2.0, 1.0, 2.0, 15.0],
        [1.5, 1.1, 0.2, 30.0],
    ]
    best = None
    for start in starts:
        result = least_squares(residual, start, bounds=([0.0, 0.0, 1e-5, 1e-4], [20.0, 20.0, 100.0, 500.0]), max_nfev=30000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))