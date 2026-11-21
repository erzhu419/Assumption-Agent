import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s):
    j0, beta, rho, c_sqrt = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * np.sqrt(u)

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'] / 0.02, r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    result = least_squares(lambda p: _law(p, u, s) - y, [3.0, 1.0, 0.70, 0.0], bounds=([0.0, 0.0, 0.05, -np.inf], [np.inf, np.inf, 0.95, np.inf]))
    _params = result.x

def predict(row):
    return float(_law(_params, row['u_now'] / 0.02, row['s_grad']))
