import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows]) / 0.1
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    start = [2.0 * max(1.0, float(np.max(y))), 1.0, 0.5, 0.1]
    result = least_squares(lambda p: _response(p, u, s) - y, start,
                           bounds=([0.0, math.ulp(1.0), 0.0, 0.0], np.inf),
                           x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_response(_params, float(row['u_now']) / 0.1, float(row['s_grad'])))