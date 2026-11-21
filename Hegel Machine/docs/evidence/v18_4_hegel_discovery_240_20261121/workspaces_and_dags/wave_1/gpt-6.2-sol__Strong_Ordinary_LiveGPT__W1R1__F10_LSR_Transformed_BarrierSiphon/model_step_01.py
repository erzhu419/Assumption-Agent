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
    branch = np.sqrt(u)

    def residual(p):
        j, beta, rho, c = p
        return j * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * branch - y

    _params = least_squares(residual, [2.0, 1.0, 0.5, 0.0], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    j, beta, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))