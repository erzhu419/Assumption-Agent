import numpy as np
from scipy.optimize import least_squares

_params = None

def _eval(p, u, s):
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) / 3.0 for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 6.0):
        result = least_squares(lambda p: _eval(p, u, s) - y, [3.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    u = float(row['u_now']) / 3.0
    return float(_eval(_params, u, float(row['s_grad'])))
