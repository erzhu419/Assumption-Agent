import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)
    local = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho * sink) * drive + c_loc * local - k_aval * np.maximum(0.0, s - s_crit) * u - y

    result = least_squares(residual, [6.5, 0.70, 0.2, 1.0, 2.59], bounds=([0.0, 0.0, -np.inf, 0.0, 2.44], [np.inf, 1.0, np.inf, np.inf, 2.78]))
    _params = result.x


def predict(row):
    vmax, rho, c_loc, k_aval, s_crit = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u)
