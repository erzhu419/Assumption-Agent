import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    u2 = u * u

    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * u2 / ((k_half + u2) * (1.0 + alpha * s)) - leak * u - y

    _params = least_squares(residual, [6.0, 1.0, 0.5, 0.3], bounds=([0.0, 1e-8, 0.0, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now'])
    u2 = math.pow(u, 2)
    return float(vmax * u2 / ((k_half + u2) * (1.0 + alpha * row['s_grad'])) - leak * u)
