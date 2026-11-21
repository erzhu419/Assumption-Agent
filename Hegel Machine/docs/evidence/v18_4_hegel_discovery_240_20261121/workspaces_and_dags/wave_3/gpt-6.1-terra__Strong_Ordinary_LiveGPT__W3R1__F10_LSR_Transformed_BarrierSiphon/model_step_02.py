import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    j0, beta, rho, c_sqrt = p
    return j0 * u * np.exp(-beta / (0.5 + u)) * (1.0 - rho * s / (1.0 + s)) + c_sqrt * np.sqrt(u)

def fit(train_rows):
    global _params
    u = np.asarray([r['u_now'] / 0.02 for r in train_rows], dtype=float)
    s = np.asarray([r['s_grad'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    best = None
    for beta in (0.2, 1.0, 3.0, 6.0):
        result = least_squares(lambda p: _model(p, u, s) - y, [3.0, beta, 0.7, 0.5], bounds=([0.0, 0.0, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), max_nfev=10000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        if best is None or result.cost < best.cost:
            best = result
    _params = best.x

def predict(row):
    u = row['u_now'] / 0.02
    return float(_model(_params, u, row['s_grad']))
