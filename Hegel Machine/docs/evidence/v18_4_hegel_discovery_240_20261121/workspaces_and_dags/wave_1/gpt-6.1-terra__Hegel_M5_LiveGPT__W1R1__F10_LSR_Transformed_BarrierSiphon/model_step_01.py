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
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return np.array([_value(r, p) for r in train_rows]) - y
    bounds = ([0.0, 0.0, 0.05, -10.0], [100.0, 20.0, 0.95, 10.0])
    candidates = [least_squares(residual, [j, b, 0.7, c], bounds=bounds, max_nfev=5000) for j in (3.0, 8.0) for b in (0.5, 2.0) for c in (-0.5, 0.5)]
    _params = min(candidates, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    return float(_value(row, _params))