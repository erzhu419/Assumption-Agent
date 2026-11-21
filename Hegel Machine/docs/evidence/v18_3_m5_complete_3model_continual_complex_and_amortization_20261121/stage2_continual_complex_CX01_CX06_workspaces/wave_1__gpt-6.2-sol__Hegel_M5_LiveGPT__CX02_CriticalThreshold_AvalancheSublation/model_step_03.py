import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _model(u, s, root_u, p):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * root_u - k_aval * u * np.maximum(0.0, s - s_crit)


def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root_u = np.sqrt(u)
    result = least_squares(lambda p: _model(u, s, root_u, p) - y,
                           x0=[6.5, 0.70, 0.2, 1.0, 1.4],
                           bounds=([1.0, 0.05, -2.0, 0.0, 0.0],
                                   [15.0, 0.95, 2.0, np.inf, np.inf]),
                           ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x


def predict(row):
    u = float(row['u_now'])
    return float(_model(u, float(row['s_grad']), math.sqrt(u), _params))
