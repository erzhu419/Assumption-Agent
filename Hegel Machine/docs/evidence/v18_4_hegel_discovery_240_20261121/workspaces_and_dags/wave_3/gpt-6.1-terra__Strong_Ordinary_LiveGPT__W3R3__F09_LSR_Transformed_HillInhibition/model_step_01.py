import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] / 0.1 for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * u**2 / (k_half + u**2) / (1.0 + alpha*s) - leak*u - y
    candidates = []
    for k in (0.2, 1.0, 4.0):
        for alpha in (0.2, 1.0, 3.0):
            result = least_squares(residual, [6.0, k, alpha, 0.1], bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=3000)
            candidates.append(result)
    _params = min(candidates, key=lambda r: np.dot(r.fun, r.fun)).x

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now']) / 0.1
    return float(vmax * u*u / (k_half + u*u) / (1.0 + alpha*float(row['s_grad'])) - leak*u)
