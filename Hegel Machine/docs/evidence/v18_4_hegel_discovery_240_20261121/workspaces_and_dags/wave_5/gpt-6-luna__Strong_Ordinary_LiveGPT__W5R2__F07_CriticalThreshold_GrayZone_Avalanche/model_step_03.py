import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base + c_loc * math.sqrt(max(u, 0.0)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _PARAMS
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    lower = np.array([0.0, 0.0, -10.0, 0.0, 2.0])
    upper = np.array([20.0, 1.0, 10.0, 10.0, 3.2])
    best = None
    for start in ([5.0, 0.5, 0.0, 1.0, 2.6], [8.0, 0.7, 0.5, 2.0, 2.5], [4.0, 0.3, -0.5, 0.5, 2.8]):
        result = least_squares(lambda p: np.asarray([_value(r, p) for r in train_rows]) - y,
                               np.asarray(start, dtype=float), bounds=(lower, upper), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _PARAMS))