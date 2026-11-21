import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root - y

    result = least_squares(residual, [4.0, 1.0, 0.70, 0.0],
                           bounds=([0.0, 0.0, 0.05, -np.inf],
                                   [np.inf, np.inf, 0.95, np.inf]))
    _params = result.x
    return _params


def predict(row):
    j0, beta, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))
