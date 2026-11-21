import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    j0, beta, rho, c_sqrt = p
    u = row['u_now'] / 3.0
    s = row['s_grad']
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - row['y'] for row in train_rows])
    candidates = []
    for beta in (0.5, 2.0, 5.0, 10.0):
        result = least_squares(residual, [3.0, beta, 0.7, 0.0], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000)
        candidates.append(result)
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_value(row, _params))
