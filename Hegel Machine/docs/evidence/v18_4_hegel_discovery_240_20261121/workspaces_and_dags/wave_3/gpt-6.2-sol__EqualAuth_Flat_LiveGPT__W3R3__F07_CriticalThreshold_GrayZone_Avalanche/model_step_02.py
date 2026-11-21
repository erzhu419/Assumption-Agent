import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    f = u / (1.2 + u)
    g = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho*g) * f + c_loc*root - k_aval*np.maximum(0.0, s-s_crit)*u - y

    _params = least_squares(residual, [7.0, 0.7, 0.3, 1.5, 2.59], bounds=([0.0, 0.0, 0.0, 0.0, 2.44], [np.inf, 1.0, np.inf, np.inf, 2.78]), gtol=1e-12).x

def predict(row) -> float:
    vmax, rho, c_loc, k_aval, s_crit = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float(vmax * (1.0 - rho*s/(1.0+s)) * u/(1.2+u) + c_loc*math.sqrt(u) - k_aval*max(0.0, s-s_crit)*u)
