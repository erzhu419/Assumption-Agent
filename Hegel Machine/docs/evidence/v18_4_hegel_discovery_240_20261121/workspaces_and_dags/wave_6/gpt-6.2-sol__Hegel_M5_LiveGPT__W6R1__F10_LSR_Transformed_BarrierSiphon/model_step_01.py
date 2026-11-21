import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 3.0 for r in train_rows])
    s = np.array([r['s_grad'] for r in train_rows])
    y = np.array([r['y'] for r in train_rows])
    g = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * g) + c * root - y

    result = least_squares(residual, [4.0, 1.0, 0.70, 0.0], bounds=([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf]))
    _params = result.x
    return _params

def predict(row):
    j0, beta, rho, c = _params
    u = row['u_now'] / 3.0
    s = row['s_grad']
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))