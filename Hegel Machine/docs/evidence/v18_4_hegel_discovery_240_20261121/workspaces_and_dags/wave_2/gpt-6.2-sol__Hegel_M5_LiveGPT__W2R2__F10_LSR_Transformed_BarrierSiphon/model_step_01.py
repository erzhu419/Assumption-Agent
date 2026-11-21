import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p, exp, sqrt):
    j0, beta, rho, c_sqrt = p
    return j0 * u * exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * sqrt(u)

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'] / 3.0, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(
        lambda p: _law(u, s, p, np.exp, np.sqrt) - y,
        x0=[6.5, 1.0, 0.70, 0.1],
        bounds=([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf]),
        ftol=1e-11, xtol=1e-11, gtol=1e-11,
    ).x

def predict(row):
    return float(_law(row['u_now'] / 3.0, row['s_grad'], _params, math.exp, math.sqrt))
