import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(p, row):
    vmax, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    H = float(row['H'])
    return vmax * (u / (1.2 + u)) * (1.0 - math.exp(-H / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-H / tau_slow))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(p, r) for r in rows]) - y
    starts = ([5.0, 0.5, 0.5, 10.0], [10.0, 1.0, 0.5, 20.0], [3.0, 0.3, 0.2, 8.0], [10.0, 0.2, 1.0, 30.0])
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=x0, bounds=([0.0, 1e-5, 0.0, 1e-4], [100.0, 100.0, 100.0, 1e4]), max_nfev=30000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.asarray(best[1], dtype=float)
    return _PARAMS.tolist()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_PARAMS, row))