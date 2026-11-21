import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        j, beta, rho, c = p
        return j * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root - y

    _params = least_squares(residual, [4.0, 1.0, 0.7, 0.3], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row):
    j, beta, rho, c = _params
    u, s = row['u_now'], row['s_grad']
    return float(j * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))