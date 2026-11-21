import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    u = float(row['u_now']) / 3.0
    s = float(row['s_grad'])
    j0, beta, rho, c = p
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(r, p) - float(r['y']) for r in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 6.0):
        result = least_squares(residual, [3.0, beta, 0.7, 0.2], bounds=([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf]), max_nfev=5000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_model(row, _params))