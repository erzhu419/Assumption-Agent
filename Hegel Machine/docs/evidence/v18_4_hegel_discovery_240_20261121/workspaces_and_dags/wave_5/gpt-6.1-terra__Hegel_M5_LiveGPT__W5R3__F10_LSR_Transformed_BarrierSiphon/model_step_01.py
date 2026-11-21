import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c_sqrt = p
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in rows])
    best = None
    for beta in (0.1, 1.0, 3.0, 8.0):
        result = least_squares(residual, [6.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))
