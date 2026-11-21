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
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho * sink) * sat + c_loc * root - k_aval * np.maximum(0.0, s - s_crit) * u - y

    _params = least_squares(residual, [7.0, 0.70, 0.30, 1.7, 2.55], bounds=([1.0, 0.05, -2.0, 0.0, 2.34], [15.0, 0.95, 2.0, np.inf, 2.78]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    vmax, rho, c_loc, k_aval, s_crit = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u)
    return float(base - k_aval * max(0.0, s - s_crit) * u)
