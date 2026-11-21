import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    j0, beta, rho, c = p
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(row, p) - float(row['y']) for row in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 8.0):
        for rho in (0.2, 0.7):
            result = least_squares(residual, [2.0, beta, rho, 0.2], bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(row, _params))