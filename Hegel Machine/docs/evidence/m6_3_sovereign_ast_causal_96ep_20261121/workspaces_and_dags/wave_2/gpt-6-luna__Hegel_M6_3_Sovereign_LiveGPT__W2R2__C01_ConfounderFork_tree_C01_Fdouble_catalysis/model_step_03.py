import math
import numpy as np
from scipy.optimize import least_squares
_params = (7.0, 1.0, 0.5)

def fit(train_rows):
    global _params
    z = np.array([float(r['z_4']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    q = (z / 1.5) ** 2
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y
    result = least_squares(residual, x0=np.array([7.0, 1.0, 0.5]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 20.0]))
    _params = tuple(float(v) for v in result.x)
    return predict

def predict(row):
    vmax, k_half, c_ctrl = _params
    q = (float(row['z_4']) / 1.5) ** 2
    return float(vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl']))