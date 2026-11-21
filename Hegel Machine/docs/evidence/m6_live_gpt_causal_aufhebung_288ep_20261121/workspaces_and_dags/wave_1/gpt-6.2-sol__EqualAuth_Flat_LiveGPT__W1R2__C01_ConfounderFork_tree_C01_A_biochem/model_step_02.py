import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.array([(row['z_2'], row['u_ctrl'], row['y']) for row in train_rows], dtype=float)
    z, u, y = data.T
    z_sq = z * z
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * z_sq / (k_half + z_sq) - c_ctrl * u - y
    _params = least_squares(residual, [8.0, 1.0, 0.2], bounds=([0.0, 1e-12, 0.0], [math.inf, math.inf, math.inf])).x

def predict(row) -> float:
    vmax, k_half, c_ctrl = _params
    z_sq = float(row['z_2']) ** 2
    return float(vmax * z_sq / (k_half + z_sq) - c_ctrl * float(row['u_ctrl']))
