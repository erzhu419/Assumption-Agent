import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, row):
    u = row['u_now'] / 0.02
    s = row['s_grad']
    j, beta, rho, c = p
    return j * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    def residual(p):
        return np.array([_model(p, r) - r['y'] for r in train_rows])
    fits = [least_squares(residual, start, bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, 1, np.inf]), max_nfev=10000) for start in ([2, 1, 0.7, 0], [4, 3, 0.7, 0], [3, 0.2, 0.5, 0.1])]
    _params = min(fits, key=lambda result: np.linalg.norm(result.fun)).x

def predict(row):
    return float(_model(_params, row))