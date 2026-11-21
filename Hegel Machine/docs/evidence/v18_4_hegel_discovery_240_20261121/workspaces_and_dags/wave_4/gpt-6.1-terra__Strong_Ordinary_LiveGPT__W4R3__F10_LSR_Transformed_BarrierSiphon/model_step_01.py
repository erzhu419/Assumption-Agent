import numpy as np
from scipy.optimize import least_squares

_params = None

def _value(row, p):
    u = row['u_now'] / 0.02
    s = row['s_grad']
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_value(r, p) - r['y'] for r in train_rows])
    best = None
    for beta in (0.2, 1.0, 3.0, 8.0):
        result = least_squares(residual, [3.0, beta, 0.7, 0.2], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_value(row, _params))
