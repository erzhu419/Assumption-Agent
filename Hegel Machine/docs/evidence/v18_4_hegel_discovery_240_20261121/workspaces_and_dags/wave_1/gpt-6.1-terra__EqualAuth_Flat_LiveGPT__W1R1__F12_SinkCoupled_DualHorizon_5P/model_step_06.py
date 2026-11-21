import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_sqrt, a_fast, tau_fast, a_slow, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * math.sqrt(max(0.0, u))
    adaptation = u * (a_fast * (-math.expm1(-h / tau_fast)) - a_slow * (-math.expm1(-h / tau_slow)))
    return base + adaptation

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    lower = [0.0, 0.0, -10.0, -10.0, 0.01, -10.0, 0.5]
    upper = [30.0, 2.0, 10.0, 10.0, 10.0, 10.0, 100.0]
    best = None
    for tf in (0.2, 1.0, 4.0):
        for ts in (5.0, 15.0, 40.0):
            p0 = [4.0, 0.7, 0.0, 1.0, tf, 2.0, ts]
            result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))
