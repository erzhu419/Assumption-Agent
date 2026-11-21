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
    results = [least_squares(residual, start, bounds=(0, np.inf), max_nfev=5000) for start in ([5, 1, 0.5, 0.1], [10, 4, 1, 0.3], [3, 0.3, 0.2, 0.05])]
    _params = min(results, key=lambda r: np.sum(r.fun**2)).x

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now']) / 0.1
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha * float(row['s_grad'])) - leak * u)
