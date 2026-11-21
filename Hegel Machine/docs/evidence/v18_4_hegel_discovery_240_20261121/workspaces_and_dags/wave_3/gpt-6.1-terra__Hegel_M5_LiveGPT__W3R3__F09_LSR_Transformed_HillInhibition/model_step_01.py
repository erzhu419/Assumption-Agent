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
    results = [least_squares(residual, x0, bounds=(0, np.inf), max_nfev=10000) for x0 in ([8, 1, 1, 0.1], [5, 0.2, 0.5, 0.05], [15, 3, 2, 0.2])]
    _params = min(results, key=lambda r: np.dot(r.fun, r.fun)).x
    return _params

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now']) / 0.1
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha * float(row['s_grad'])) - leak * u)
