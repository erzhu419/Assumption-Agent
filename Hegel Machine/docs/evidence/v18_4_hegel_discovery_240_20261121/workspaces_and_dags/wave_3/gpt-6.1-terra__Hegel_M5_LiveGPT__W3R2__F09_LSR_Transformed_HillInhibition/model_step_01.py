import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, k_half, alpha, leak = p
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        return _model(p, u, s) - y
    best = None
    for k in (0.1, 1.0, 5.0):
        for alpha in (0.1, 1.0, 3.0):
            result = least_squares(residual, [max(1.0, float(np.max(y)) * 2), k, alpha, 0.1], bounds=([0.0, 1e-10, 0.0, 0.0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=5000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))