import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    j0, beta, rho, c = p
    un = u / 3.0
    return j0 * un * np.exp(-beta / (0.5 + un)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(un)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for beta in (0.1, 1.0, 3.0, 6.0):
        result = least_squares(lambda p: _model(p, u, s) - y,
                               [6.5, beta, 0.7, 0.1],
                               bounds=([0.0, 0.0, 0.0, -np.inf],
                                       [np.inf, np.inf, 1.0, np.inf]),
                               max_nfev=5000)
        if best is None or np.linalg.norm(result.fun) < np.linalg.norm(best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_model(_params, float(row['u_now']), float(row['s_grad'])))