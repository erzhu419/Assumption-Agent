import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_TAU = 1.0

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) / _TAU for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root - y

    _params = least_squares(residual, [5.0, 1.0, 0.70, 0.0], bounds=([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf])).x
    return _params

def predict(row) -> float:
    j0, beta, rho, c = _params
    u = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))