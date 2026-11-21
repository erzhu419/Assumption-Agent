import math
import numpy as np
from scipy.optimize import least_squares
_params = (8.0, 1.0, 0.2)

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) / 1.5 for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    q = x * x
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.2]), bounds=([0.0, 1e-8, -10.0], [100.0, 100.0, 10.0]))
    _params = tuple(float(v) for v in result.x)
    return predict

def predict(row):
    vmax, k_half, c_ctrl = _params
    q = (float(row['x_ind']) / 1.5) ** 2
    return float(vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl']))