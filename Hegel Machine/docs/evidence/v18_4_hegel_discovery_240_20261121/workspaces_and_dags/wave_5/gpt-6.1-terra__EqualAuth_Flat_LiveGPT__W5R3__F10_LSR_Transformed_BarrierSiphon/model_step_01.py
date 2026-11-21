import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    j0, beta, rho, c = p
    z = u / 0.02
    return j0 * z * np.exp(-beta / (0.5 + z)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(z)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for beta in (0.2, 1.0, 3.0, 8.0):
        result = least_squares(lambda p: _model(p, u, s) - y,
                               [2.0, beta, 0.5, 0.1],
                               bounds=([0.0, 0.0, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]),
                               max_nfev=10000)
        if best is None or np.sum(result.fun**2) < np.sum(best.fun**2):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_model(_params, row['u_now'], row['s_grad']))