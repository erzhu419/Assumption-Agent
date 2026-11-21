import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] / 0.1 for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    u2 = u * u

    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * (u2 / (k_half + u2)) / (1.0 + alpha * s) - leak * u - y

    _params = least_squares(residual, [max(1.0, float(np.max(y))), 1.0, 0.5, 0.1], bounds=([0.0, 1e-12, 0.0, 0.0], np.inf)).x

def predict(row) -> float:
    vmax, k_half, alpha, leak = _params
    u = row['u_now'] / 0.1
    u2 = math.pow(u, 2)
    return float(vmax * (u2 / (k_half + u2)) / (1.0 + alpha * row['s_grad']) - leak * u)
