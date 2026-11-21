import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(row, p):
    j0, beta, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(row, p) - float(row['y']) for row in train_rows])
    best = None
    for beta in (0.1, 1.0, 3.0, 8.0):
        for c in (-1.0, 0.0, 1.0):
            result = least_squares(residual, [6.0, beta, 0.7, c], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(row, _params))