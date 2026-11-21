import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    a, b, c, fast, slow, tau_fast, tau_slow = p
    base = (a - b * s / (1.0 + s)) * u / (1.2 + u) + c * np.sqrt(max(u, 0.0))
    adaptation = u * (fast * (-np.expm1(-h / tau_fast)) - slow * (-np.expm1(-h / tau_slow)))
    return float(base + adaptation)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(p, r) - float(r['y']) for r in rows])
    lower = [0.0, 0.0, 0.0, 0.0, 0.0, 0.05, 1.0]
    upper = [20.0, 20.0, 20.0, 20.0, 20.0, 15.0, 100.0]
    best = None
    for tf in (0.5, 2.0, 5.0):
        for ts in (8.0, 20.0, 40.0):
            start = [1.5, 0.8, 0.5, 1.0, 2.0, tf, ts]
            result = least_squares(residual, start, bounds=(lower, upper), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return _value(_params, row)
