import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _model(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u))
    return base * (1.0 - math.exp(-h / tau_fast)) - amp_adapt * u * (1.0 - math.exp(-h / tau_slow))

def fit(train_rows):
    global _PARAMS
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        return np.asarray([_model(r, p) for r in train_rows]) - y
    lower = np.array([0.01, 0.0, 0.01, 0.0, 1.0])
    upper = np.array([30.0, 0.999, 5.0, 10.0, 200.0])
    best = None
    for start in ([6.0, 0.7, 0.3, 0.5, 15.0], [6.0, 0.7, 1.0, 0.2, 30.0], [4.0, 0.5, 0.1, 1.0, 10.0]):
        result = least_squares(residual, np.asarray(start), bounds=(lower, upper), max_nfev=20000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _PARAMS = best.x
    return _PARAMS.copy()

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    return float(_model(row, _PARAMS))