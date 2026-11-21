import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, k_half, alpha, leak = p
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 0.1 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for start in ([8, 1, 0.5, 0.1], [5, 0.5, 1, 0.01], [15, 3, 2, 0.3]):
        res = least_squares(lambda p: _model(p, u, s) - y, start, bounds=([0, 1e-9, 0, 0], [np.inf, np.inf, np.inf, np.inf]), max_nfev=5000)
        if best is None or np.linalg.norm(res.fun) < np.linalg.norm(best.fun):
            best = res
    _params = best.x

def predict(row):
    return float(_model(_params, row['u_now'] / 0.1, row['s_grad']))