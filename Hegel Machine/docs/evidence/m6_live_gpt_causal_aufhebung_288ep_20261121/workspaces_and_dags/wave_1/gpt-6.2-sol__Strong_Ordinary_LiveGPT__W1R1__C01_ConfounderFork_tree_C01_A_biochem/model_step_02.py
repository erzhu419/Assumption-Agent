import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    z, u, y = np.asarray([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float).T
    q = z**2

    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y

    _params = least_squares(residual, [max(1.0, float(np.max(y))), 1.0, 0.3], bounds=([0.0, 1e-9, 0.0], np.inf)).x

def predict(row) -> float:
    vmax, k_half, c_ctrl = _params
    q = math.pow(row['z_2'], 2)
    return float(vmax * q / (k_half + q) - c_ctrl * row['u_ctrl'])
