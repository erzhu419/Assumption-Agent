import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([float(r['u_now']) / 4.0 for r in train_rows])
    s = np.array([float(r['s_grad']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)
    local = np.sqrt(u)

    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * sink) * drive + c * local - y

    _params = least_squares(residual, [6.5, 0.70, 0.14], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row):
    vmax, rho, c = _params
    u = float(row['u_now']) / 4.0
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))
