import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c = p
    u = float(row['u_now']) / 0.02
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    results = [least_squares(residual, x0, bounds=([0.0, 0.0, 0.0, -10.0], [100.0, 100.0, 1.0, 10.0]), max_nfev=10000) for x0 in ([6.0, 1.0, 0.7, 0.1], [3.0, 3.0, 0.7, 0.5], [10.0, 0.2, 0.5, 0.0])]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_value(row, _params))
