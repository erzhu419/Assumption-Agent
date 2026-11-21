import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    fast = -math.expm1(-h / tau_fast)
    slow = -math.expm1(-h / tau_slow)
    return vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) * fast - amp_adapt * u * slow

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.01], [30.0, 1.0, 20.0, 10.0, 500.0])
    best = None
    for initial in ([6.5, 0.7, 0.5, 1.0, 10.0], [6.5, 0.7, 1.0, 2.0, 25.0], [6.5, 0.7, 0.2, 0.5, 5.0]):
        result = least_squares(residual, initial, bounds=bounds, max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    if _params is None:
        raise RuntimeError('fit must be called before predict')
    return float(_value(row, _params))