import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, khalf, alpha, leak = p
        return vmax * u**2 / (khalf + u**2) / (1 + alpha*s) - leak*u - y
    best = None
    for k in (0.3, 1.0, 3.0):
        for a in (0.2, 1.0, 3.0):
            result = least_squares(residual, [6.0, k, a, 0.1], bounds=([0, 1e-9, 0, 0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    vmax, khalf, alpha, leak = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * u*u / (khalf + u*u) / (1 + alpha*s) - leak*u)
