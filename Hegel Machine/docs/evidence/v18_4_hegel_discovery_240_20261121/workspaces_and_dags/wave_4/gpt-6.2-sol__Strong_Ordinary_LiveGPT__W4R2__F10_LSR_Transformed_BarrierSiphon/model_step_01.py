import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    sink = s / (1.0 + s)
    root = np.sqrt(u)
    def residual(p):
        j0, beta, rho, c_sqrt = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c_sqrt * root - y
    _params = least_squares(residual, [4.0, 1.0, 0.7, 0.0], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf])).x

def predict(row):
    j0, beta, rho, c_sqrt = _params
    u, s = row['u_now'], row['s_grad']
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u))