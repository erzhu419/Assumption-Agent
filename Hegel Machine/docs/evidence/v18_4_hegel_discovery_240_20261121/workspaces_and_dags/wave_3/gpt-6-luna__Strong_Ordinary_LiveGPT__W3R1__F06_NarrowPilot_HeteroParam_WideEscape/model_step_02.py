import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u, s

def _predict_params(row, p):
    u, s = _features(row)
    j0, beta_bar, rho, c_sqrt = p
    un = u  # characteristic tau scale is 1.0
    return j0 * un * math.exp(-beta_bar / (0.5 + un)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(0.0, un))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_predict_params(r, p) for r in rows]) - y
    starts = [
        [5.0, 0.5, 0.5, 0.5],
        [10.0, 1.0, 0.7, 0.2],
        [3.0, 0.1, 0.3, 1.0],
        [20.0, 2.0, 0.8, 0.1],
    ]
    best = None
    for x0 in starts:
        result = least_squares(residual, x0=x0, bounds=([0.0, 0.0, 0.0, -20.0], [1000.0, 100.0, 1.0, 20.0]), max_nfev=20000)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_predict_params(row, _PARAMS))