import math
import numpy as np
from scipy.optimize import least_squares

_TAU = 2.5
_params = None

def fit(train_rows):
    global _params
    z, s, y = np.asarray([(float(r['u_now']) / _TAU, float(r['s_grad']), float(r['y'])) for r in train_rows], dtype=float).T
    q = np.square(z)

    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * q / (k_half + q) / (1.0 + alpha * s) - leak * z - y

    start = [2.0 * max(1.0, float(np.max(y))), 1.0, 1.0, 0.2]
    _params = least_squares(residual, start, bounds=([0.0, 1e-10, 0.0, 0.0], np.inf)).x

def predict(row) -> float:
    vmax, k_half, alpha, leak = _params
    z = float(row['u_now']) / _TAU
    q = math.pow(z, 2)
    s = float(row['s_grad'])
    return float(vmax * q / (k_half + q) / (1.0 + alpha * s) - leak * z)
