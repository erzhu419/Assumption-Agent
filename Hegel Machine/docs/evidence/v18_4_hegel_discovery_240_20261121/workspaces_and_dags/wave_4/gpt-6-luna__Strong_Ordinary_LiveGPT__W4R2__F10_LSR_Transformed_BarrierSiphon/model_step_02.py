import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    j0, beta_bar, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(0.0, u))

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows], dtype=float)
    lo = np.array([0.0, 0.0, 0.0, -np.inf])
    hi = np.array([np.inf, np.inf, 1.0, np.inf])
    starts = [np.array([3.0, 0.5, 0.5, 0.0]), np.array([5.0, 1.0, 0.7, 0.5]), np.array([2.0, 0.1, 0.2, -0.5])]
    best = None
    for x0 in starts:
        res = least_squares(lambda p: np.asarray([_value(p, r) for r in rows]) - y, x0, bounds=(lo, hi), max_nfev=20000)
        if best is None or np.sum(res.fun ** 2) < np.sum(best.fun ** 2):
            best = res
    _params = best.x.copy()
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(_params, row))