import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _model(p, u, s) - y
    starts = [[2.0, 1.0, 0.5, 0.5], [4.0, 3.0, 0.8, 0.1], [1.0, 0.2, 0.3, 1.0]]
    results = [least_squares(residual, p, bounds=([0, 0, 0, 0], [np.inf, np.inf, 1, np.inf]), max_nfev=10000) for p in starts]
    _params = min(results, key=lambda r: np.sum(r.fun ** 2)).x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))