import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, vmax, rho, c_loc, k_aval, s_crit):
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return base + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _law(u, s, *p) - y,
                           x0=[6.0, 0.70, 0.4, 1.5, 2.55],
                           bounds=([0.0, 0.0, -np.inf, 0.0, 0.0],
                                   [np.inf, 1.0, np.inf, np.inf, np.inf])).x
    return _params

def predict(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, rho, c_loc, k_aval, s_crit = map(float, _params)
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
    return float(base + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u)
