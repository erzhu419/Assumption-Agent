import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(u, s, p):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * u2 / (k_half + u2) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows]) / 0.1
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    result = least_squares(lambda p: _response(u, s, p) - y,
                           [6.0, 2.0, 1.0, 0.1],
                           bounds=([0.0, 1e-12, 0.0, 0.0], math.inf),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    return float(_response(float(row['u_now']) / 0.1, float(row['s_grad']), _params))
