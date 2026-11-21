import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _values(p, u, s):
    vmax, rho, c_loc, k_aval, s_crit = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * np.sqrt(u) - k_aval * np.maximum(0.0, s - s_crit) * u

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    result = least_squares(lambda p: _values(p, u, s) - y, [6.5, 0.7, 0.2, 2.0, 2.6], bounds=([1.0, 0.05, -2.0, 0.0, 2.5], [15.0, 0.95, 2.0, 20.0, 2.68]), max_nfev=10000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
    _params = result.x

def predict(row):
    return float(_values(_params, float(row['u_now']), float(row['s_grad'])))
