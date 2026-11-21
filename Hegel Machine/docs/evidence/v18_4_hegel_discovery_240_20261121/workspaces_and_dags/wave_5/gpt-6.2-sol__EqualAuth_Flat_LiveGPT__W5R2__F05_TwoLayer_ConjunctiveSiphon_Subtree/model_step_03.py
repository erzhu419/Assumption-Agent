import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)
    local = np.sqrt(u)
    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * sink) * drive + c_loc * local - y
    _params = least_squares(residual, [5.0, 0.5, 0.5], bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf])).x

def predict(row) -> float:
    u = float(row['u_now'])
    s = float(row['s_grad'])
    vmax, rho, c_loc = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u))