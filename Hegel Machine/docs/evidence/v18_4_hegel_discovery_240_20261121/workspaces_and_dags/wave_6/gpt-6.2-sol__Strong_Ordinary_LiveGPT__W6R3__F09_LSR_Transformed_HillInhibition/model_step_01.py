import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    u2 = u * u

    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * u2 / ((k_half + u2) * (1.0 + alpha * s)) - leak * u - y

    _params = least_squares(residual, [2.0 * np.max(y), 1.0, 1.0, 0.2], bounds=([0.0, 1e-9, 0.0, 0.0], np.inf), x_scale='jac').x

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = float(row['u_now'])
    u2 = math.pow(u, 2)
    s = float(row['s_grad'])
    return float(vmax * u2 / ((k_half + u2) * (1.0 + alpha * s)) - leak * u)
