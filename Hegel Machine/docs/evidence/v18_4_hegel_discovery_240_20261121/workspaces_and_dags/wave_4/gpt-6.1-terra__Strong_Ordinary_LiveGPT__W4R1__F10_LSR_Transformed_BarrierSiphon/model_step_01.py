import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        j, beta, rho, c = p
        return j * u * np.exp(-beta / (0.5 + u)) * (1 - rho * s / (1 + s)) + c * np.sqrt(u) - y
    candidates = [least_squares(residual, [j, b, r, c], bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, 1, np.inf]), max_nfev=5000) for j, b, r, c in [(2, 1, 0.7, 0.2), (4, 2, 0.7, 0.5), (2, 0.2, 0.5, 0)]]
    _params = min(candidates, key=lambda result: np.dot(result.fun, result.fun)).x

def predict(row):
    j, beta, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(j * u * np.exp(-beta / (0.5 + u)) * (1 - rho * s / (1 + s)) + c * np.sqrt(u))