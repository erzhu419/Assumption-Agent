import math
import numpy as np
from scipy.optimize import least_squares

TAU = 0.1
_params = None

def _law(p, u, s):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * u2 / (k_half + u2) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] / TAU for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    def residual(p):
        return _law(p, u, s) - y
    result = least_squares(residual, [8.0, 1.0, 1.0, 0.2], bounds=([0.0, 1e-12, 0.0, 0.0], [math.inf] * 4), x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row) -> float:
    return float(_law(_params, row['u_now'] / TAU, row['s_grad']))