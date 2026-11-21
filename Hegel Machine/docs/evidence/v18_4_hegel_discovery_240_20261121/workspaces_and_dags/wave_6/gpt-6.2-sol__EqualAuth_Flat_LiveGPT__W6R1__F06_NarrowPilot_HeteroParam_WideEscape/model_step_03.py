import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    grad = s / (1.0 + s)
    local = np.sqrt(u)
    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * grad) * base + c_loc * local - y
    _params = least_squares(residual, [6.0, 0.8, 0.4], bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf])).x

def predict(row):
    vmax, rho, c_loc = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u))