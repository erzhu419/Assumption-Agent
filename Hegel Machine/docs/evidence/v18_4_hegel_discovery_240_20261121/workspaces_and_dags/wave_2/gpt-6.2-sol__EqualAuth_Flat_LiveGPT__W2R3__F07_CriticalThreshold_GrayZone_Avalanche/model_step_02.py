import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, vmax, rho, c_loc, k_aval, s_crit):
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    _params = least_squares(lambda p: _law(u, s, *p) - y, [6.0, 0.5, 0.5, 2.0, 2.6], bounds=([0.0, 0.0, 0.0, 0.0, 2.44], [np.inf, 1.0, np.inf, np.inf, 2.78]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    return float(_law(float(row['u_now']), float(row['s_grad']), *_params))
