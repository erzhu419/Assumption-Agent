import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(rows, p):
    j0, beta, rho, c = p
    u = np.asarray([r['u_now'] for r in rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in rows], dtype=float)
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    target = np.asarray([r['y'] for r in train_rows], dtype=float)
    best = None
    for beta in (0.2, 1.0, 3.0, 6.0):
        for rho in (0.2, 0.7):
            result = least_squares(lambda p: _values(train_rows, p) - target,
                                   [3.0, beta, rho, 0.5],
                                   bounds=([0.0, 0.0, 0.0, 0.0],
                                           [np.inf, np.inf, 1.0, np.inf]),
                                   max_nfev=5000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
                best = result
    _params = best.x

def predict(row):
    return float(_values([row], _params)[0])
