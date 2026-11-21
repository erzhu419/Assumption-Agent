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
    y = np.array([float(r['y']) for r in rows])
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    # Multiple positive initializations help resolve correlated fast/slow timescales.
    for start in ([1.0, 1.0, 0.5, 10.0], [1.5, 0.5, 0.8, 20.0], [2.0, 2.0, 1.0, 30.0], [0.8, 0.2, 0.4, 8.0]):
        result = least_squares(residual, np.array(start, dtype=float), bounds=([0.0, 1e-5, 0.0, 1e-5], [20.0, 1e4, 20.0, 1e5]), max_nfev=20000)
        score = float(np.dot(result.fun, result.fun))
        if best is None or score < best[0]:
            best = (score, result.x)
    _PARAMS = np.array(best[1], dtype=float)
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _PARAMS))