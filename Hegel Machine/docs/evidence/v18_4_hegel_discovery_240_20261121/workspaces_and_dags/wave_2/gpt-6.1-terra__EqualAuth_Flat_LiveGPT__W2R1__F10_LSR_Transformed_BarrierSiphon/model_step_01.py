import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c_sqrt = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    results = [least_squares(residual, [j, b, 0.5, 0.1], max_nfev=10000) for j, b in [(2.0, 0.5), (3.0, 2.0), (5.0, 5.0)]]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    return float(_value(row, _params))