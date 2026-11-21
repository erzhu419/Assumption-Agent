import numpy as np
from scipy.optimize import least_squares

_params = None


def _law(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha_inh * s)) - c_leak * u


def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    result = least_squares(lambda p: _law(p, u, s) - y,
                           [7.0, 1.0, 1.0, 0.1],
                           bounds=([0.0, 1e-10, 0.0, 0.0], np.inf),
                           x_scale='jac', ftol=1e-11, xtol=1e-11, gtol=1e-11)
    _params = result.x
    return _params


def predict(row):
    return float(_law(_params, float(row['u_now']), float(row['s_grad'])))