import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float)
    s = data[:, 0] ** 2
    u, y = data[:, 1], data[:, 2]

    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * s / (k_half + s) - c_ctrl * u - y

    _params = least_squares(residual, [max(1.0, float(y.max())), 1.0, 0.1], bounds=([0.0, 1e-12, 0.0], np.inf)).x

def predict(row) -> float:
    vmax, k_half, c_ctrl = _params
    s = math.pow(float(row['z_2']), 2)
    return float(vmax * s / (k_half + s) - c_ctrl * float(row['u_ctrl']))
