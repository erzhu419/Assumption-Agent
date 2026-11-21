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
    result = least_squares(residual, [8.0, 1.0, 0.5, 0.1], bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=10000)
    _params = result.x

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = row['u_now'] / 0.1
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha * row['s_grad']) - leak * u)
