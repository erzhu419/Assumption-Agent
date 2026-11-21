import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    data = np.array([(r['u_now'] / 2.5, r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    result = least_squares(lambda p: _law(u, s, p) - y, [2.0 * max(1.0, float(np.max(y))), 1.0, 1.0, 0.2], bounds=([0.0, 1e-10, 0.0, 0.0], np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    return float(_law(row['u_now'] / 2.5, row['s_grad'], _params))
