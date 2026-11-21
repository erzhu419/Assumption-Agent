import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    u = row['u_now']
    s = row['s_grad']
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(max(0.0, u)) - k_aval * max(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - r['y'] for r in rows])
    best = None
    for threshold in (2.35, 2.5, 2.65, 2.75):
        result = least_squares(residual, [6.5, 0.7, 0.2, 2.0, threshold], bounds=([0.0, 0.0, -5.0, 0.0, 0.0], [30.0, 1.0, 5.0, 20.0, 5.0]), max_nfev=10000, ftol=1e-13, xtol=1e-13, gtol=1e-13)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))