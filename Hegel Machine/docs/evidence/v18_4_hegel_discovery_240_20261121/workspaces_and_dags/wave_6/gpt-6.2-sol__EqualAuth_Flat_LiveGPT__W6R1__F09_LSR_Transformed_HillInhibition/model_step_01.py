import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha_inh * s)) - c_leak * u

def fit(train_rows):
    global _params
    data = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    _params = least_squares(lambda p: _law(u, s, p) - y, [4.0, 1.0, 1.0, 0.1], bounds=([0.0, 1e-10, 0.0, 0.0], np.inf), x_scale='jac').x
    return _params

def predict(row) -> float:
    return float(_law(float(row['u_now']), float(row['s_grad']), _params))