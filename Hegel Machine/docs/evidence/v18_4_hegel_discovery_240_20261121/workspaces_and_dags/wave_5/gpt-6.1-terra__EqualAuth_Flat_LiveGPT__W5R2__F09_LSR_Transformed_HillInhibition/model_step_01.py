import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 2.5 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, k, alpha, leak = p
        return vmax * u**2 / (k + u**2) / (1.0 + alpha*s) - leak*u - y
    best = None
    for k in (0.1, 1.0, 10.0):
        for alpha in (0.1, 1.0, 5.0):
            result = least_squares(residual, [max(y)*2, k, alpha, 0.1], bounds=([0, 1e-9, 0, 0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=10000)
            if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                best = result
    _params = best.x

def predict(row):
    vmax, k, alpha, leak = _params
    u = float(row['u_now']) / 2.5
    return float(vmax*u*u/(k+u*u)/(1.0+alpha*float(row['s_grad'])) - leak*u)
