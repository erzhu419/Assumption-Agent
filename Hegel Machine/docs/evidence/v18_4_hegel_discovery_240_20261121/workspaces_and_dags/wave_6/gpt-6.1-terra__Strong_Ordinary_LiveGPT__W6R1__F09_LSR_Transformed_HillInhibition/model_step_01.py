import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, k_half, alpha, leak = p
    return vmax * u**2 / (k_half + u**2) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    best = None
    for k in (0.3, 1.0, 3.0):
        for alpha in (0.3, 1.0, 3.0):
            result = least_squares(lambda p: _model(p, u, s) - y,
                                   [5.0, k, alpha, 0.1],
                                   bounds=([0.0, 1e-9, 0.0, 0.0],
                                           [np.inf, np.inf, np.inf, np.inf]),
                                   max_nfev=5000)
            if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
                best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))
