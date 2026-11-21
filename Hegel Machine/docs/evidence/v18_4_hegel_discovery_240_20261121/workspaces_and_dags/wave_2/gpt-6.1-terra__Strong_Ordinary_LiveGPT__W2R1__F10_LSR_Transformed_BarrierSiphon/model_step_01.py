import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    j0, beta, rho, c = p
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(row, p) - float(row['y']) for row in train_rows])
    candidates = []
    for beta in (0.1, 0.5, 1.0, 2.0, 4.0):
        result = least_squares(residual, [3.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 20.0, 1.0, np.inf]), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        candidates.append(result)
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    return float(_value(row, _params))