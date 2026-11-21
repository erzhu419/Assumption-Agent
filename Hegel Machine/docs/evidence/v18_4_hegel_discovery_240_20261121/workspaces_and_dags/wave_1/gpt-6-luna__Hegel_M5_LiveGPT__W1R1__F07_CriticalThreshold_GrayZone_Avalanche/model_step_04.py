import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _predict_params(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = max(0.0, float(row['u_now']))
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    branch = c_loc * math.sqrt(u)
    avalanche = k_aval * max(0.0, s - s_crit) * u
    return base + branch - avalanche

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    lower = np.array([1.0, 0.05, -2.0, 0.0, 0.0])
    upper = np.array([15.0, 0.95, 2.0, 10.0, 5.0])
    best = None
    for start in ([6.5, 0.70, 0.1, 1.0, 2.5], [6.5, 0.70, 0.1, 3.0, 2.8], [8.0, 0.65, -0.2, 5.0, 3.0]):
        result = least_squares(lambda p: np.asarray([_predict_params(r, p) for r in train_rows]) - y,
                               np.asarray(start, dtype=float), bounds=(lower, upper), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _PARAMS = best.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_predict_params(row, _PARAMS))