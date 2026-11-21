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
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * sink) * drive + c_loc * local - y

    _params = least_squares(residual, [7.0, 0.5, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, rho, c_loc = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u))