import numpy as np
from scipy.optimize import least_squares

_params = None

def _response(p, u, s):
    vmax, k_half, alpha_inh, c_leak = p
    u2 = u * u
    return vmax * u2 / ((k_half + u2) * (1.0 + alpha_inh * s)) - c_leak * u

def fit(train_rows):
    global _params
    u, s, y = np.array([(float(r['u_now']) / 0.1, float(r['s_grad']), float(r['y'])) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _response(p, u, s) - y, [2.0 * max(1.0, float(np.max(y))), 1.0, 0.5, 0.1], bounds=([0.0, 1e-10, 0.0, 0.0], np.inf), x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_response(_params, float(row['u_now']) / 0.1, float(row['s_grad'])))