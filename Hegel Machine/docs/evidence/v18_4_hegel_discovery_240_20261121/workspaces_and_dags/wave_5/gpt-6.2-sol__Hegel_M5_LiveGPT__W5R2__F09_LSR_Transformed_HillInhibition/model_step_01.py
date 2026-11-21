import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    q = u * u
    return vmax * (q / (k_half + q)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    u = np.asarray([float(r['u_now']) / 2.5 for r in train_rows])
    s = np.asarray([float(r['s_grad']) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    start = [max(1.0, 2.0 * float(np.max(y))), 0.5, 1.0, 0.4]
    result = least_squares(lambda p: _law(p, u, s) - y, start,
                           bounds=([0.0, math.ulp(1.0), 0.0, 0.0], np.inf),
                           x_scale='jac')
    _params = result.x
    return _params

def predict(row):
    return float(_law(_params, float(row['u_now']) / 2.5, float(row['s_grad'])))