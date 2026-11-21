import numpy as np
from scipy.optimize import least_squares

_params = None

def _evaluate(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    result = least_squares(lambda p: _evaluate(p, u, s) - y,
                           x0=[6.0, 1.0, 1.0, 0.2],
                           bounds=([0.0, 1e-12, 0.0, 0.0], np.inf))
    _params = result.x

def predict(row) -> float:
    return float(_evaluate(_params, row['u_now'], row['s_grad']))
