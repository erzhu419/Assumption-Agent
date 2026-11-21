import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(float(r['u_now']) / 2.5, float(r['s_grad']), float(r['y'])) for r in train_rows], dtype=float).T
    u2 = u * u

    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * (u2 / (k_half + u2)) / (1.0 + alpha * s) - leak * u - y

    _params = least_squares(residual, x0=[max(1.0, 2.0 * float(np.max(y))), 1.0, 0.7, 0.15], bounds=([0.0, 1e-8, 0.0, 0.0], np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now']) / 2.5
    u2 = math.pow(u, 2)
    return float(vmax * (u2 / (k_half + u2)) / (1.0 + alpha * float(row['s_grad'])) - leak * u)
