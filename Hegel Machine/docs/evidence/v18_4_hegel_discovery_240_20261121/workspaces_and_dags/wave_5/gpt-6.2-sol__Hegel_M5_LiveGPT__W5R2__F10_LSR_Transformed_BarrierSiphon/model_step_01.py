import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, q, root_u, p):
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * q) + c * root_u

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'] / 0.02, r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    q = s / (1.0 + s)
    root_u = np.sqrt(u)
    result = least_squares(lambda p: _law(u, q, root_u, p) - y,
                           x0=[4.0, 1.0, 0.70, 0.0],
                           bounds=([0.0, 0.0, 0.05, -np.inf],
                                   [np.inf, np.inf, 0.95, np.inf]))
    _params = result.x

def predict(row) -> float:
    u = row['u_now'] / 0.02
    s = row['s_grad']
    return float(_law(u, s / (1.0 + s), math.sqrt(u), _params))
