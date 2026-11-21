import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c = p
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    fits = [least_squares(residual, start, bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000) for start in ([3.0, 1.0, 0.5, 0.2], [5.0, 3.0, 0.8, 0.1], [1.0, 0.2, 0.2, 0.5])]
    _params = min(fits, key=lambda result: np.dot(result.fun, result.fun)).x
    return _params

def predict(row):
    return float(_value(row, _params))