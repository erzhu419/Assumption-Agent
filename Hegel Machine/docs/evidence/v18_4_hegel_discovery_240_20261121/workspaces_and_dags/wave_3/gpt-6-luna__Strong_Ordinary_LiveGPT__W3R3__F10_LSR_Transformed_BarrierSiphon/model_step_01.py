import math
import numpy as np
from scipy.optimize import least_squares

def _features(row):
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    barrier = u * math.exp(-0.0)  # keeps feature construction explicit
    return u, s, barrier

def _value(row, p):
    j0, beta, rho, c_sqrt = p
    u, s, _ = _features(row)
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    best = None
    # Multiple starts address the nonlinear parameter tradeoff with only five rows.
    for start in ([2.0, 0.5, 0.5, 0.0], [10.0, 1.0, 0.7, -0.2], [1.0, 0.1, 0.2, 0.2]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, -5.0, -10.0], [1e4, 20.0, 5.0, 10.0]), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    return best.x

def predict(row):
    return float(_value(row, _FITTED_PARAMS))

_FITTED_PARAMS = np.array([1.0, 0.5, 0.5, 0.0], dtype=float)

def _fit_and_bind(train_rows):
    global _FITTED_PARAMS
    _FITTED_PARAMS = fit(train_rows)
    return _FITTED_PARAMS
