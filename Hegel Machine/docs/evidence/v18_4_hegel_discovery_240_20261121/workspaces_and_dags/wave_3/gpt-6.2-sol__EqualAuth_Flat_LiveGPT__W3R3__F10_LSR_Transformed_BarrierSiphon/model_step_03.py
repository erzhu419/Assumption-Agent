import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_TAU = 0.02

def _law(p, u, s):
    j0, beta_bar, rho, c_sqrt = p
    return j0 * u * np.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * np.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) / _TAU for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    result = least_squares(lambda p: _law(p, u, s) - y, [3.0, 1.0, 0.5, 0.3], bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row) -> float:
    j0, beta_bar, rho, c_sqrt = _params
    u = float(row['u_now']) / _TAU
    s = float(row['s_grad'])
    return float(j0 * u * math.exp(-beta_bar / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * math.sqrt(u))
