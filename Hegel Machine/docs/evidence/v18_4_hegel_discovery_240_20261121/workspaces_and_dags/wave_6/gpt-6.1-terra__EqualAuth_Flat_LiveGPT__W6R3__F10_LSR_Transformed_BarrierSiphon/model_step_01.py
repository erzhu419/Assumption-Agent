import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(row, p):
    j0, beta, rho, c = p
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_evaluate(row, p) - float(row['y']) for row in train_rows])
    best = None
    for beta in (0.1, 1.0, 4.0):
        for rho in (0.2, 0.7):
            result = least_squares(residual, [2.0, beta, rho, 0.5], bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]), max_nfev=5000)
            if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_evaluate(row, _params))