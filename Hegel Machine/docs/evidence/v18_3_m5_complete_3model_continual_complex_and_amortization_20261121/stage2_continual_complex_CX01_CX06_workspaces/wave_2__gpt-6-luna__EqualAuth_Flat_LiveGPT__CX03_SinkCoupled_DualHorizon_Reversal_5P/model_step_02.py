import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _predict_with(row, p):
    a_fast, tau_fast, a_slow, tau_slow = p
    h = max(0.0, float(row['H']))
    u = float(row['u_now'])
    return u * (a_fast * (1.0 - math.exp(-h / tau_fast)) - a_slow * (1.0 - math.exp(-h / tau_slow)))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_with(r, p) for r in rows]) - y
    best = None
    # Enforce fast/slow ordering through separate bounded ranges.
    for start in ([1.0, 0.3, 0.8, 10.0], [1.5, 1.0, 1.0, 20.0], [2.0, 2.0, 1.5, 40.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.03, 0.0, 2.1], [10.0, 2.0, 10.0, 200.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_with(row, _params))