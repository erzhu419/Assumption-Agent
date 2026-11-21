import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho * sink) * drive + c_loc * root - k_aval * u * np.maximum(0.0, s - s_crit) - y

    _params = least_squares(residual, [7.0, 0.7, 0.3, 1.5, 2.6], bounds=([0.0, 0.0, -np.inf, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    vmax, rho, c_loc, k_aval, s_crit = _params
    u, s = row['u_now'], row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u) - k_aval * u * max(0.0, s - s_crit))