import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    q = (x / 1.5) ** 2
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y
    result = least_squares(residual, x0=[8.0, 1.0, 0.5], bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _params = tuple(float(v) for v in result.x)
    return predict

def predict(row):
    vmax, k_half, c_ctrl = _params
    q = (float(row['x_ind']) / 1.5) ** 2
    return float(vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl']))