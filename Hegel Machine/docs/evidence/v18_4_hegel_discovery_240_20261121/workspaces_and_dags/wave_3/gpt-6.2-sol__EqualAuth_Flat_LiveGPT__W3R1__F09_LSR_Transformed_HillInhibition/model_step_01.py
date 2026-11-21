import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, params):
    vmax, k_half, alpha_inh, c_leak = params
    q = u * u
    return vmax * q / ((k_half + q) * (1.0 + alpha_inh * s)) - c_leak * u

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'] / 0.1, r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    result = least_squares(lambda p: _law(u, s, p) - y, x0=[6.0, 2.0, 0.5, 0.3], bounds=([0.0, 1e-12, 0.0, 0.0], np.inf))
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_law(row['u_now'] / 0.1, row['s_grad'], _params))