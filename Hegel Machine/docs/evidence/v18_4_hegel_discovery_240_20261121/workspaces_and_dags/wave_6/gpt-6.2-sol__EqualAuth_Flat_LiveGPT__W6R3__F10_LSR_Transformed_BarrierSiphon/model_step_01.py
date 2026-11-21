import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    sink = s / (1.0 + s)
    root = np.sqrt(u)
    barrier = 1.0 / (0.5 + u)

    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta * barrier) * (1.0 - rho * sink) + c * root - y

    _params = least_squares(residual, [2.0, 1.0, 0.5, 0.2], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf])).x

def predict(row):
    j0, beta, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))