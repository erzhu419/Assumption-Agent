import math
import numpy as np
from scipy.optimize import least_squares

_params = None
_TAU_SCALE = 1.0

def fit(train_rows):
    global _params
    data = np.array([[r['u_now'], r['s_grad'], r['y']] for r in train_rows], dtype=float)
    u, s, y = data.T
    u = u / _TAU_SCALE
    u2 = u * u

    def residual(p):
        vmax, k_half, alpha, c_leak = p
        return vmax * u2 / (k_half + u2) / (1.0 + alpha * s) - c_leak * u - y

    _params = least_squares(residual, [5.0, 1.0, 0.5, 0.1], bounds=([0.0, 1e-12, 0.0, 0.0], np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    vmax, k_half, alpha, c_leak = _params
    u = float(row['u_now']) / _TAU_SCALE
    u2 = math.pow(u, 2)
    return float(vmax * u2 / (k_half + u2) / (1.0 + alpha * float(row['s_grad'])) - c_leak * u)
