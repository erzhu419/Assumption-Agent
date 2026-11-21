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
        return np.array([_model(row, p) - float(row['y']) for row in train_rows])
    best = None
    for start in ([3.0, 1.0, 0.5, 0.5], [5.0, 3.0, 0.8, 0.1], [2.0, 0.2, 0.3, 1.0]):
        result = least_squares(residual, start, bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(row, _params))