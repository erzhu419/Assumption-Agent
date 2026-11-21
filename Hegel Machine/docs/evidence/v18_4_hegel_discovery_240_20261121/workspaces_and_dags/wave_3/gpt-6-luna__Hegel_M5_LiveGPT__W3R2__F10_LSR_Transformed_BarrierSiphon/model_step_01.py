import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return u, s

def _value(p, row):
    j0, beta_bar, rho, c_sqrt = p
    u, s = _features(row)
    return j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(max(u, 0.0))

def fit(train_rows):
    global _params
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(p, r) for r in train_rows]) - y
    x0 = np.array([6.5, 1.0, 0.70, 0.5], dtype=float)
    lo = np.array([0.0, 0.0, 0.05, -20.0])
    hi = np.array([100.0, 20.0, 0.95, 20.0])
    result = least_squares(residual, x0, bounds=(lo, hi), max_nfev=20000)
    _params = result.x
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    return float(_value(_params, row))