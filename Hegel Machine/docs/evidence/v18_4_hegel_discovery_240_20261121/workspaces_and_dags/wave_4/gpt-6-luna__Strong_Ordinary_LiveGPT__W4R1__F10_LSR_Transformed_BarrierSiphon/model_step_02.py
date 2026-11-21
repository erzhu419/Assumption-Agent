import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta_bar, rho, c_sqrt = p
    u = max(0.0, float(row['u_now']))
    s = max(0.0, float(row['s_grad']))
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.asarray([float(r['y']) for r in rows])
    def residual(p):
        return np.asarray([_value(r, p) for r in rows]) - y
    starts = ([2.0, 0.5, 0.7, 0.0], [1.0, 1.0, 0.5, 0.2], [4.0, 0.1, 0.9, -0.2])
    best = None
    for x0 in starts:
        result = least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 20.0, 1.0, 10.0]), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x.copy()
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_value(row, _params))