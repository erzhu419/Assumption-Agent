import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    amp_fast, amp_slow, tau_fast, tau_slow = np.exp(p)
    h = float(row['H'])
    return float(row['u_now']) * (amp_fast * (-np.expm1(-h / tau_fast)) - amp_slow * (-np.expm1(-h / tau_slow)))

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(p, row) - float(row['y']) for row in train_rows])
    bounds = (np.log([1e-5, 1e-5, 1e-3, 0.1]), np.log([100.0, 100.0, 100.0, 1000.0]))
    best = None
    for fast in (0.3, 1.0, 3.0):
        for slow in (5.0, 15.0, 40.0):
            result = least_squares(residual, np.log([2.0, 3.0, fast, slow]), bounds=bounds, max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(_params, row))
