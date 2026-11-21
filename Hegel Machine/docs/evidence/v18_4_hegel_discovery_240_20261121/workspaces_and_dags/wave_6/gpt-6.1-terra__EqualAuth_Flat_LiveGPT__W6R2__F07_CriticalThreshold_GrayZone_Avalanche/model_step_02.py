import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(max(0.0, u)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    bounds = ([0.0, 0.0, -10.0, 0.0, 2.50], [100.0, 2.0, 20.0, 100.0, 2.68])
    best = None
    for threshold in (2.51, 2.59, 2.67):
        result = least_squares(residual, [5.0, 0.3, 1.0, 2.0, threshold], bounds=bounds, max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))