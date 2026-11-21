import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([[r['u_now'], r['s_grad'], r['y']] for r in train_rows], dtype=float)
    u = data[:, 0] / 0.02
    sink = data[:, 1] / (1.0 + data[:, 1])
    root = np.sqrt(u)
    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root - data[:, 2]
    _params = least_squares(residual, [3.0, 1.0, 0.70, 0.0], bounds=([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf])).x
    return _params

def predict(row) -> float:
    j0, beta, rho, c = _params
    u = row['u_now'] / 0.02
    s = row['s_grad']
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))