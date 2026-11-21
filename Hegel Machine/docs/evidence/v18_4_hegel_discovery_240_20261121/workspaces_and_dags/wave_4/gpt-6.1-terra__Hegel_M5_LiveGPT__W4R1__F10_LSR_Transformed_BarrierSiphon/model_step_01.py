import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, rows):
    j0, beta, rho, c = p
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    y = np.array([r['y'] for r in train_rows], dtype=float)
    best = None
    for beta in (0.2, 1.0, 3.0, 6.0):
        result = least_squares(lambda p: _values(p, train_rows) - y,
                               [5.0, beta, 0.7, 0.0],
                               bounds=([0.0, 0.0, 0.0, -np.inf],
                                       [np.inf, np.inf, 1.0, np.inf]),
                               max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
        if best is None or np.sum(result.fun ** 2) < np.sum(best.fun ** 2):
            best = result
    _params = best.x

def predict(row):
    return float(_values(_params, [row])[0])
