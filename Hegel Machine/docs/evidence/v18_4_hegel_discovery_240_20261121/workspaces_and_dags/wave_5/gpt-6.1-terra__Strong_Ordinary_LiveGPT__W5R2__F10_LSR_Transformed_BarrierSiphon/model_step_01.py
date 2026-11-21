import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, s):
    j, beta, rho, c = p
    return j * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 0.02 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for beta in (0.5, 2.0, 5.0, 10.0):
        result = least_squares(lambda p: _values(p, u, s) - y, [3.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    return float(_values(_params, row['u_now'] / 0.02, row['s_grad']))
