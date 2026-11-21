import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - float(r['y']) for r in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 7.0):
        for c in (0.0, 0.5, 1.5):
            result = least_squares(residual, [3.0, beta, 0.7, c], bounds=([0.0, 0.0, 0.0, -20.0], [100.0, 30.0, 1.0, 20.0]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))