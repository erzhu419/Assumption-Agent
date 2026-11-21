import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s):
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    a = np.array([(r['u_now'] / 0.02, r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = a.T
    result = least_squares(lambda p: _law(p, u, s) - y, [3.0, 1.0, 0.7, -0.2], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    u = row['u_now'] / 0.02
    s = row['s_grad']
    j0, beta, rho, c = _params
    return float(j0 * u * math.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * math.sqrt(u))