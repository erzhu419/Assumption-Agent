import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(p, u, s):
    vmax, rho, c_loc, k_aval, s_crit = p
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            + c_loc * np.sqrt(u) - k_aval * u * np.maximum(0.0, s - s_crit))

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _model(p, u, s) - y,
                           [6.0, 0.7, 0.3, 1.7, 2.6],
                           bounds=([0.0, 0.0, -np.inf, 0.0, 2.44],
                                   [np.inf, 1.0, np.inf, np.inf, 2.78]))
    _params = result.x

def predict(row) -> float:
    u, s = row['u_now'], row['s_grad']
    vmax, rho, c_loc, k_aval, s_crit = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
                 + c_loc * math.sqrt(u) - k_aval * u * max(0.0, s - s_crit))