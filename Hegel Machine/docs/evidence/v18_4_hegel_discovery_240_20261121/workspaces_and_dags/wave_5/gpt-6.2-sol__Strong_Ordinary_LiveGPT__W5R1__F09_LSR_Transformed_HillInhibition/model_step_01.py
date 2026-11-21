import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, k_half, alpha, leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha * s) - leak * u

def fit(train_rows):
    global _params
    data = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    result = least_squares(lambda p: _law(u, s, p) - y, x0=[5.0, 1.0, 1.0, 0.2], bounds=([0.0, 1e-9, 0.0, 0.0], np.inf), x_scale='jac')
    _params = result.x
    return _params

def predict(row):
    return float(_law(float(row['u_now']), float(row['s_grad']), _params))
