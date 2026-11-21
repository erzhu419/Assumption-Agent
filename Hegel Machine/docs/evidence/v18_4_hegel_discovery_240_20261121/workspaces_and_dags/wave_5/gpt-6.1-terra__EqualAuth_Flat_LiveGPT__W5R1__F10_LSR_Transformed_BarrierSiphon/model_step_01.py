import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c = p
    u = row['u_now'] / 0.02
    s = row['s_grad']
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - r['y'] for r in train_rows])
    candidates = [least_squares(residual, [j, b, r, c], bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, 1, np.inf]), max_nfev=5000) for j, b, r, c in [(1, 1, 0.5, 0.1), (2, 3, 0.8, 0.0), (3, 0.1, 0.3, 0.1)]]
    _params = min(candidates, key=lambda result: np.dot(result.fun, result.fun)).x
    return _params

def predict(row):
    return float(_value(row, _params))