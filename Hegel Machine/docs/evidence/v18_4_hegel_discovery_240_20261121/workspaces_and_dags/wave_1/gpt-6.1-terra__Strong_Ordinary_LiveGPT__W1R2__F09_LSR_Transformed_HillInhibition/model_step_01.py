import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 2.5 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * u**2 / (k_half + u**2) / (1.0 + alpha*s) - leak*u - y
    best = None
    for start in ([8, 1, 1, 0.1], [5, 0.3, 0.5, 0.1], [15, 3, 2, 0.5]):
        result = least_squares(residual, start, bounds=([0, 1e-8, 0, 0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now']) / 2.5
    return float(vmax * u*u / (k_half + u*u) / (1.0 + alpha*float(row['s_grad'])) - leak*u)
