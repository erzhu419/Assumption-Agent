import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u - y
    fits = [least_squares(residual, start, bounds=([0, 1e-9, 0, 0], [np.inf]*4), max_nfev=10000) for start in ([5, 1, 1, 0.1], [10, 2, 0.5, 0.3], [3, 0.2, 0.1, 0.01])]
    _params = min(fits, key=lambda result: np.sum(result.fun**2)).x
    return _params

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u)
