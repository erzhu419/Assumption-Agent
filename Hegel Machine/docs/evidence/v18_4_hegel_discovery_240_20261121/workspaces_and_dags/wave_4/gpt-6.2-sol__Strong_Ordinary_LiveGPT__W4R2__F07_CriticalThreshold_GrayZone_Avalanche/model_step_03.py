import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1 - rho * s / (1 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * u * np.maximum(0.0, s - s_crit)

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    result = least_squares(lambda p: _law(p, u, s) - y, [7.0, 0.7, 0.4, 1.5, 2.6], bounds=([0.0, 0.0, -np.inf, 0.0, 2.44], [np.inf, 1.0, np.inf, np.inf, 2.78]), ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x
    return _params

def predict(row):
    return float(_law(_params, row['u_now'], row['s_grad']))