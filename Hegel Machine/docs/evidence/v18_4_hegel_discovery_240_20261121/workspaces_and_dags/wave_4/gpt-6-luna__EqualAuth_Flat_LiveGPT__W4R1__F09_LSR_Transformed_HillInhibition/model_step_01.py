import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, k_half, alpha_inh, c_leak = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (u * u / (k_half + u * u)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    y = np.array([float(r['y']) for r in train_rows])
    def residual(z):
        p = np.exp(z)
        return np.array([_value(r, p) for r in train_rows]) - y
    starts = [np.log([3.0, 1.0, 0.5, 0.05]), np.log([5.0, 2.0, 1.0, 0.1])]
    best = None
    for z0 in starts:
        result = least_squares(residual, z0, max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = np.exp(best.x)
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    return float(_value(row, _params))
