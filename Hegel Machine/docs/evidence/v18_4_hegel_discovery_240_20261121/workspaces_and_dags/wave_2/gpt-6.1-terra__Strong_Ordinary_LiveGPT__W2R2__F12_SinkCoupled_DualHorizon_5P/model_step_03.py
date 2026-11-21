import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, tau_fast, amp_adapt, tau_slow = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    h = float(row['H'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) * (-math.expm1(-h / tau_fast)) - amp_adapt * u * (-math.expm1(-h / tau_slow))

def fit(train_rows):
    global _params
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.asarray([_value(r, p) for r in train_rows]) - y
    bounds = ([0.0, 0.0, 0.001, 0.0, 0.1], [100.0, 1.5, 20.0, 20.0, 1000.0])
    best = None
    for tf in (0.1, 0.5, 1.5):
        for ts in (5.0, 20.0, 80.0):
            result = least_squares(residual, [5.0, 0.7, tf, 1.0, ts], bounds=bounds, max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))
