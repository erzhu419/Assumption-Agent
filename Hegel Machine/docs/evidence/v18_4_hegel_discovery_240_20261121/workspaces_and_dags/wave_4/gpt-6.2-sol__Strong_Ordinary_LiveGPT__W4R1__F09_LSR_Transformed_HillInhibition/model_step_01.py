import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    u2 = np.square(u)

    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * (u2 / (k_half + u2)) / (1.0 + alpha*s) - leak*u - y

    _params = least_squares(residual, [8.0, 1.0, 1.0, 0.3], bounds=([0.0, 1e-9, 0.0, 0.0], np.inf), ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now'])
    u2 = math.pow(u, 2)
    return float(vmax * (u2 / (k_half + u2)) / (1.0 + alpha*float(row['s_grad'])) - leak*u)
