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
    best = None
    for k in (0.2, 1.0, 4.0):
        for alpha in (0.1, 1.0, 3.0):
            result = least_squares(residual, [8.0, k, alpha, 0.1], bounds=(0.0, np.inf), max_nfev=5000)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x
    return _params

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now']) / 0.1
    return float(vmax * u**2 / (k_half + u**2) / (1.0 + alpha * float(row['s_grad'])) - leak * u)
