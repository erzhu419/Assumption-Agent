import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1 - rho * s / (1 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * np.maximum(0, s - s_crit) * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _law(u, s, p) - y, [6.5, 0.70, 0.3, 1.5, 2.6], bounds=([0, 0, -np.inf, 0, 2.44], [np.inf, 1, np.inf, np.inf, 2.78]))
    _params = result.x

def predict(row):
    u, s = float(row['u_now']), float(row['s_grad'])
    vmax, rho, c_loc, k_aval, s_crit = _params
    return float(vmax * (1 - rho * s / (1 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u) - k_aval * max(0, s - s_crit) * u)
