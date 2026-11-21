import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(u, s, response, local, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * response + c_loc * local - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    response = u / (1.2 + u)
    local = np.sqrt(u)
    result = least_squares(
        lambda p: _evaluate(u, s, response, local, p) - y,
        [6.5, 0.70, 0.0, 1.0, 2.59],
        bounds=([0.0, 0.0, -np.inf, 0.0, 2.50], [np.inf, 1.0, np.inf, np.inf, 2.68]),
        ftol=1e-12, xtol=1e-12, gtol=1e-12,
    )
    _params = result.x

def predict(row) -> float:
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(_evaluate(u, s, u / (1.2 + u), math.sqrt(u), _params))