import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    sat = u / (1.2 + u)
    grad = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho * grad) * sat + c_loc * root - k_aval * u * np.maximum(0.0, s - s_crit) - y

    _params = least_squares(residual, [6.0, 0.7, 0.3, 1.5, 2.59], bounds=([0.0, 0.0, 0.0, 0.0, 2.50], [np.inf, 1.0, np.inf, np.inf, 2.68])).x

def predict(row) -> float:
    vmax, rho, c_loc, k_aval, s_crit = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u)
    return float(base - k_aval * u * max(0.0, s - s_crit))