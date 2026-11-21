import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_TAU = 3.0

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u = data[:, 0] / _TAU
    sink = data[:, 1] / (1.0 + data[:, 1])
    root = np.sqrt(u)
    y = data[:, 2]

    def residual(p):
        j, beta, rho, c = p
        return j * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root - y

    result = least_squares(residual, [3.0, 1.0, 0.5, 0.0], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), x_scale='jac')
    _params = result.x
    return _params

def predict(row) -> float:
    j, beta, rho, c = _params
    u = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return float(j * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))