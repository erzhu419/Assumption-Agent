import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float)
    z, u, y = data.T
    q = z ** 2

    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * q / (k_half + q) - c_ctrl * u - y

    _params = least_squares(residual, [max(1.0, float(y.max())), 1.0, 0.3], bounds=([0.0, 1e-12, 0.0], [math.inf, math.inf, math.inf])).x

def predict(row):
    vmax, k_half, c_ctrl = _params
    q = float(row['z_2']) ** 2
    return float(vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl']))
