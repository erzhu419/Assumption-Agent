import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * (u2 / (k_half + u2)) / (1.0 + alpha_inh * s) - c_leak * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _law(p, u, s) - y,
                           [2.0 * max(y), 1.0, 0.5, 0.2],
                           bounds=([0.0, 1e-10, 0.0, 0.0], np.inf),
                           ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x

def predict(row) -> float:
    return float(_law(_params, row['u_now'], row['s_grad']))
