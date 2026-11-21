import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    j0, beta, rho, c = p
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    y = np.asarray [float(r['y']) for r in train_rows]
    def residual(p):
        return np.asarray([_model(p, r) for r in train_rows]) - y
    best = None
    for beta in (0.1, 1.0, 3.0, 8.0):
        for rho in (0.3, 0.7, 1.0):
            result = least_squares(residual, [3.0, beta, rho, 0.1], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, 30.0, 2.0, np.inf]), max_nfev=10000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, row))