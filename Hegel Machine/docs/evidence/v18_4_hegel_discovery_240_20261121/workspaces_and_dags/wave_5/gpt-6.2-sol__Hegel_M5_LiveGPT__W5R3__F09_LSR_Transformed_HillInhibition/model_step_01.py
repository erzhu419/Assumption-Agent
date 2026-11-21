import numpy as np
from scipy.optimize import least_squares

_params = None

def _model(u, s, p):
    vmax, k_half, alpha_inh, c_leak = p
    t = u * u
    return vmax * (t / (k_half + t)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'] / 0.1, r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    start = [2.0 * max(1.0, float(np.max(y))), 1.0, 1.0, 0.1]
    result = least_squares(lambda p: _model(u, s, p) - y, start, bounds=([0.0, 1e-12, 0.0, 0.0], np.inf), x_scale='jac')
    _params = result.x
    return _params

def predict(row) -> float:
    return float(_model(row['u_now'] / 0.1, row['s_grad'], _params))
