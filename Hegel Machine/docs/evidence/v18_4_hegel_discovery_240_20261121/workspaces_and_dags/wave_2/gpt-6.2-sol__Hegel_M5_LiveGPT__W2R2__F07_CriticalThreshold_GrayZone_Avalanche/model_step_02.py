import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, sqrt_u, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    return (vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u)
            + c_loc * sqrt_u - k_aval * np.maximum(0.0, s - s_crit) * u)

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    sqrt_u = np.sqrt(u)
    result = least_squares(lambda p: _law(u, s, sqrt_u, p) - y,
                           [6.5, 0.7, 0.2, 1.0, 2.59],
                           bounds=([1.0, 0.05, -2.0, 0.0, 2.44],
                                   [15.0, 0.95, 2.0, np.inf, 2.78]))
    _params = result.x
    return _params

def predict(row) -> float:
    u = float(row['u_now'])
    return float(_law(u, float(row['s_grad']), math.sqrt(u), _params))
