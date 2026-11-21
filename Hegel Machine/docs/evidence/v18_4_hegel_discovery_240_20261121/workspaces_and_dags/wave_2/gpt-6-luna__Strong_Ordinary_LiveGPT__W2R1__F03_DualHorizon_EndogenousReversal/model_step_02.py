import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    amp_fast, tau_fast, amp_slow, tau_slow = p
    h = float(row['H'])
    u = float(row['u_now'])
    return u * (amp_fast * (1.0 - math.exp(-h / tau_fast)) - amp_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    lower = np.array([0.0, 1e-5, 0.0, 1e-5])
    upper = np.array([100.0, 1000.0, 100.0, 1000.0])
    starts = ([1.0, 0.5, 1.0, 15.0], [2.0, 1.0, 2.0, 30.0], [1.0, 0.2, 0.5, 10.0])
    best = None
    for start in starts:
        result = least_squares(lambda p: np.asarray([_value(r, p) for r in rows]) - y,
                               np.asarray(start), bounds=(lower, upper), max_nfev=20000,
                               x_scale='jac')
        score = float(np.sum(result.fun ** 2))
        if best is None or score < best[0]:
            best = (score, result.x)
    _params = best[1]
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))