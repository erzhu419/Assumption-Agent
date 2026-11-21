import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c, k, crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u) - k * max(0.0, s - crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in rows]) - y
    best = None
    for crit in (1.1, 1.3, 1.5, 1.7):
        result = least_squares(residual, [6.5, 0.7, 0.15, 2.0, crit], bounds=([1.0, 0.05, -2.0, 0.0, 0.0], [15.0, 0.95, 2.0, 20.0, 2.34]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_value(row, _params))