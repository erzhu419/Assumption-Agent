import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([float(r['u_now']) for r in rows])
    s = np.array([float(r['s_grad']) for r in rows])
    y = np.array([float(r['y']) for r in rows])
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)
    branch = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * sink) * drive + c_loc * branch - y

    _params = least_squares(residual, [6.5, 0.70, 0.15], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row) -> float:
    vmax, rho, c_loc = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))