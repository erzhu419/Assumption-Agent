import math
import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 0.1
_params = None

def _response(u, s, p):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    u = u / TAU_SCALE
    result = least_squares(
        lambda p: _response(u, s, p) - y,
        x0=[2.0 * max(1.0, float(np.max(y))), 1.0, 0.5, 0.1],
        bounds=([0.0, 1e-12, 0.0, 0.0], math.inf),
        x_scale='jac',
    )
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_response(row['u_now'] / TAU_SCALE, row['s_grad'], _params))