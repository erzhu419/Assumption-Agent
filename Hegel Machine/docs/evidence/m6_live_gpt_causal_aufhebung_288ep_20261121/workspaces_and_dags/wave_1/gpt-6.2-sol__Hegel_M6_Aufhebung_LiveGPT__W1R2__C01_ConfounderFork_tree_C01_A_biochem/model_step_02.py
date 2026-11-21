import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float)
    q = np.square(data[:, 0])
    u, y = data[:, 1], data[:, 2]

    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y

    _params = least_squares(residual, [8.0, 1.5, 0.2], bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    vmax, k_half, c_ctrl = _params
    q = math.pow(float(row['z_2']), 2)
    return float(vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl']))
