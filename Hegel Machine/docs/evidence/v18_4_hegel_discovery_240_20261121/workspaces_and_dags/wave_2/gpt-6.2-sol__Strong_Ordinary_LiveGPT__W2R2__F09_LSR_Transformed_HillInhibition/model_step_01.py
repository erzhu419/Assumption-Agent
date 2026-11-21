import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def _law(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * u2 / (k_half + u2) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) / TAU_SCALE for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    initial = [2.0 * max(1.0, float(np.max(y))), 1.0, 1.0, 0.1]
    result = least_squares(lambda p: _law(p, u, s) - y, initial,
                           bounds=([0.0, 1e-8, 0.0, 0.0], np.inf),
                           x_scale='jac')
    _params = result.x

def predict(row) -> float:
    return float(_law(_params, float(row['u_now']) / TAU_SCALE, float(row['s_grad'])))
