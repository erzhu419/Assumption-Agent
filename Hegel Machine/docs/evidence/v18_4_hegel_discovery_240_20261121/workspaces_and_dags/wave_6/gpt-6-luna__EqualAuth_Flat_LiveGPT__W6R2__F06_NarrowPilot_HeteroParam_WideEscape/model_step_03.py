import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = max(1e-8, float(row['u_now']))
    s = float(row['s_grad'])
    return u, s

def _value(row, p):
    u, s = _features(row)
    a, k, c, d, b = p
    return b + a * (1.0 - math.exp(-k * u)) + c * math.sqrt(u) + d * (s - 1.0) * u

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    starts = [
        [4.0, 0.8, 1.0, -0.1, 0.0],
        [8.0, 0.3, 0.3, -0.1, 0.0],
        [2.0, 1.5, 1.5, -0.05, 0.0],
    ]
    best = None
    for p0 in starts:
        res = least_squares(residual, p0, bounds=([-100, 1e-6, -100, -10, -100], [100, 10, 100, 10, 100]), max_nfev=20000)
        if best is None or np.dot(res.fun, res.fun) < np.dot(best.fun, best.fun):
            best = res
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))