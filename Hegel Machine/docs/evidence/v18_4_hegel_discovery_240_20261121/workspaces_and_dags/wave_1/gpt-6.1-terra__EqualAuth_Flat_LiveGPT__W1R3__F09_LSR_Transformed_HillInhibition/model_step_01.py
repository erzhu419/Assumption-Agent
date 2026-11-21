import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 0.1 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u - y
    candidates = []
    for k in (0.1, 1.0, 10.0):
        for a in (0.1, 1.0, 5.0):
            result = least_squares(residual, [max(y.max(), 0.1) * (1 + a), k, a, 0.1], bounds=([0, 1e-10, 0, 0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            candidates.append(result)
    _params = min(candidates, key=lambda r: np.sum(r.fun**2)).x
    return _params

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now']) / 0.1
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha * float(row['s_grad'])) - leak * u)
