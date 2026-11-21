import math
import numpy as np
from scipy.optimize import least_squares

def _value(p, row):
    amp_fast, amp_slow, tau_fast, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    starts = ([1.5, 0.8, 0.3, 15.0], [2.0, 1.0, 0.5, 25.0], [1.0, 0.5, 0.2, 10.0])
    best = None
    for initial in starts:
        result = least_squares(
            lambda p: np.asarray([_value(p, r) for r in rows]) - y,
            x0=np.asarray(initial, dtype=float),
            bounds=([0.0, 0.0, 1e-6, 1e-6], [100.0, 100.0, 1e4, 1e4]),
            max_nfev=20000
        )
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    return float(_value(fit(_TRAIN_ROWS), row))