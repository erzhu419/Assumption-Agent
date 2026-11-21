import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    vmax, rho, c, k, crit = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(max(0.0, u)) - k * max(0.0, s - crit) * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    best = None
    for crit in (2.3, 2.5, 2.7):
        result = least_squares(residual, [6.5, 0.7, 0.1, 1.5, crit], bounds=([1.0, 0.05, -2.0, 0.0, 0.0], [15.0, 0.95, 2.0, 10.0, 5.0]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))