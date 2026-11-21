import math
import numpy as np
from scipy.optimize import least_squares

_tau = 1.0
_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'] / _tau, r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    u2 = u * u

    def residual(p):
        vmax, k_half, alpha, leak = p
        return vmax * u2 / (k_half + u2) / (1.0 + alpha * s) - leak * u - y

    _params = least_squares(residual, [7.0, 0.8, 0.5, 0.1], bounds=([0.0, 1e-10, 0.0, 0.0], math.inf)).x
    return _params

def predict(row):
    vmax, k_half, alpha, leak = _params
    u = row['u_now'] / _tau
    u2 = u * u
    return float(vmax * u2 / (k_half + u2) / (1.0 + alpha * row['s_grad']) - leak * u)
