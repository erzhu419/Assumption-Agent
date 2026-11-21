import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 2.5 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, khalf, alpha, leak = p
        return vmax * u**2 / (khalf + u**2) / (1.0 + alpha * s) - leak * u - y
    best = None
    for k in (0.2, 1.0, 4.0):
        for a in (0.2, 1.0, 3.0):
            result = least_squares(residual, [max(y)*2, k, a, 0.1], bounds=([0, 1e-8, 0, 0], [1000, 1000, 1000, 1000]), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    vmax, khalf, alpha, leak = _params
    u = float(row['u_now']) / 2.5
    s = float(row['s_grad'])
    return float(vmax * u**2 / (khalf + u**2) / (1.0 + alpha*s) - leak*u)
