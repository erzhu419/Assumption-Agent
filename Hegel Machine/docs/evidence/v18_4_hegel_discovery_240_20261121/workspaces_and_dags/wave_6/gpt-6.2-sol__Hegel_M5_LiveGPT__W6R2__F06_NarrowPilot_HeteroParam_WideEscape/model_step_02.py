import math
import numpy as np
from scipy.optimize import least_squares


def fit(train_rows):
    global params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)
    local = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * sink) * drive + c_loc * local - y

    params = least_squares(residual, [6.5, 0.5, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x


def predict(row) -> float:
    vmax, rho, c_loc = params
    u = row['u_now']
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * (u / (1.2 + u)) + c_loc * math.sqrt(u))