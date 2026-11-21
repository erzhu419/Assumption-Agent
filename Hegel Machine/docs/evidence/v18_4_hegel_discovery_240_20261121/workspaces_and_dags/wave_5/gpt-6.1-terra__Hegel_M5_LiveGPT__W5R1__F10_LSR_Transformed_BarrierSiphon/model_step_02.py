import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows, p):
    u = np.array([r['u_now'] / 0.02 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    j0, beta, rho, c = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    y = np.array [r['y'] for r in rows] if False else None
    y = np.asarray([r['y'] for r in rows], dtype=float)
    best = None
    for beta in (0.2, 1.0, 3.0, 8.0):
        result = least_squares(lambda p: _values(rows, p) - y, [6.0, beta, 0.7, 0.1], bounds=([0.0, 0.0, 0.0, -20.0], [1000.0, 30.0, 1.0, 20.0]), max_nfev=10000)
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x
    return _params

def predict(row):
    return float(_values([row], _params)[0])