import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_TAU = 1.0

def _response(u, s, params):
    vmax, k_half, alpha, leak = params
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / _TAU for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    _params = least_squares(
        lambda p: _response(u, s, p) - y,
        x0=[2.0 * math.fabs(float(np.max(y))) + 1.0, 1.0, 0.5, 0.3],
        bounds=([0.0, 1e-10, 0.0, 0.0], np.inf),
        x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11,
    ).x

def predict(row) -> float:
    return float(_response(row['u_now'] / _TAU, row['s_grad'], _params))
