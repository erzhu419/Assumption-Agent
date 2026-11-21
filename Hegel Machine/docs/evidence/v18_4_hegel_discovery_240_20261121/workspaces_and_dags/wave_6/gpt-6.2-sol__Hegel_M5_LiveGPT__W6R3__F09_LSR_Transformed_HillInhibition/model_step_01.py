import numpy as np
from scipy.optimize import least_squares

TAU_SCALE = 1.0
_params = None

def _law(u, s, p):
    vmax, k_half, alpha, c_leak = p
    q = u * u
    return vmax * q / ((k_half + q) * (1.0 + alpha * s)) - c_leak * u

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    u = u / TAU_SCALE
    _params = least_squares(lambda p: _law(u, s, p) - y,
                            [2.0 * np.max(y), 1.0, 1.0, 0.1],
                            bounds=([0.0, 1e-10, 0.0, 0.0], np.inf),
                            ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    return float(_law(row['u_now'] / TAU_SCALE, row['s_grad'], _params))