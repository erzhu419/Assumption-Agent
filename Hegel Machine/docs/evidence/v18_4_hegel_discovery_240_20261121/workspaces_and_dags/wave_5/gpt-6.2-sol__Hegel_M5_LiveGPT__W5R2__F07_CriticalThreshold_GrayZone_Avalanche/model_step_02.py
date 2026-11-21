import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _law(p, u, s, drive, local):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * drive + c_loc * local - k_aval * np.maximum(0.0, s - s_crit) * u


def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    local = np.sqrt(u)
    result = least_squares(
        lambda p: _law(p, u, s, drive, local) - y,
        x0=[6.5, 0.7, 0.15, 1.0, 2.59],
        bounds=([1.0, 0.05, -2.0, 0.0, 2.50], [15.0, 0.95, 2.0, np.inf, 2.68]),
    )
    _params = result.x
    return _params


def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(_law(_params, u, s, u / (1.2 + u), math.sqrt(u)))
