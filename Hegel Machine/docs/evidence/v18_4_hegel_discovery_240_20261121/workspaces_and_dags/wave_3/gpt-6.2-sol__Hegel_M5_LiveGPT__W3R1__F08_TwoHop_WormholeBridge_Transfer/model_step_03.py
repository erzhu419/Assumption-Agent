import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s):
    a, k, b, ell, c, d = p
    return (a * u / (k + u) + b * u / (ell + u)) * (1.0 - c * s / (d + s))

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _law(p, u, s) - y,
                           [4.0, 2.5, 5.0, 0.75, 0.6, 1.0],
                           bounds=([0, 1e-6, 0, 1e-6, 0, 1e-6],
                                   [np.inf, np.inf, np.inf, np.inf, 1, np.inf]),
                           x_scale='jac', max_nfev=2000)
    _params = result.x

def predict(row):
    a, k, b, ell, c, d = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    transfer = math.fsum((a * u / (k + u), b * u / (ell + u)))
    return float(transfer * (1.0 - c * s / (d + s)))