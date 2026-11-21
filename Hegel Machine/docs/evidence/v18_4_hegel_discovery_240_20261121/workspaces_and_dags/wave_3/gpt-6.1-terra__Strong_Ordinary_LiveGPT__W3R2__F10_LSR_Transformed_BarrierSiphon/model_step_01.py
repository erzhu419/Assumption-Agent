import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 3.0 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        j, beta, rho, c = p
        return j * u * np.exp(-beta / (0.5 + u)) * (1 - rho * s / (1 + s)) + c * np.sqrt(u) - y
    fits = [least_squares(residual, [j, beta, rho, c], bounds=([0, 0, 0, -np.inf], [np.inf, np.inf, 1, np.inf]), max_nfev=10000) for j, beta, rho, c in [(3, 1, 0.7, 0), (5, 3, 0.7, 0.5), (2, 0.3, 0.5, -0.5)]]
    _params = min(fits, key=lambda f: np.sum(f.fun ** 2)).x

def predict(row):
    j, beta, rho, c = _params
    u = row['u_now'] / 3.0
    s = row['s_grad']
    return float(j * u * np.exp(-beta / (0.5 + u)) * (1 - rho * s / (1 + s)) + c * np.sqrt(u))