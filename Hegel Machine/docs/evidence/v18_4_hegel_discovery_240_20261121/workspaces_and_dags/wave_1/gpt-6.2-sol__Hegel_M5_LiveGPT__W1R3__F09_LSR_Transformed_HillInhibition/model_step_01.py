import math
import numpy as np
from scipy.optimize import least_squares

TAU = 0.1
_params = None

def _law(u, s, p):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha * s)) - leak * u

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    u = u / TAU
    def residual(p):
        return _law(u, s, p) - y
    start = [2.0 * max(1.0, float(np.max(y))), math.pow(float(np.median(u)), 2), 0.5, 0.1]
    result = least_squares(residual, start, bounds=([0.0, 1e-9, 0.0, 0.0], np.inf), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_law(float(row['u_now']) / TAU, float(row['s_grad']), _params))
