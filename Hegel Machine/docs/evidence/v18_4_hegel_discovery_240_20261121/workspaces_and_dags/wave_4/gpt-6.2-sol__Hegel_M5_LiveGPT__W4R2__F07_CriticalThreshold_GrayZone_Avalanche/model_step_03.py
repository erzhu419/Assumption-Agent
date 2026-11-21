import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s, root_u):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * root_u - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root_u = np.sqrt(u)
    result = least_squares(lambda p: _law(p, u, s, root_u) - y,
                           [6.5, 0.70, 0.25, 1.0, 2.6],
                           bounds=([1.0, 0.05, -2.0, 0.0, 2.44],
                                   [15.0, 0.95, 2.0, np.inf, 2.78]),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(_law(_params, u, s, math.sqrt(u)))
