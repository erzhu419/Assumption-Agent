import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] / 3.0 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    root_u = np.sqrt(u)
    sink = s / (1.0 + s)

    def residual(p):
        j0, beta, rho, c = p
        return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * sink) + c * root_u - y

    _params = least_squares(residual, [3.0, 1.0, 0.5, 0.0], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    j0, beta, rho, c = _params
    u = row['u_now'] / 3.0
    s = row['s_grad']
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))
