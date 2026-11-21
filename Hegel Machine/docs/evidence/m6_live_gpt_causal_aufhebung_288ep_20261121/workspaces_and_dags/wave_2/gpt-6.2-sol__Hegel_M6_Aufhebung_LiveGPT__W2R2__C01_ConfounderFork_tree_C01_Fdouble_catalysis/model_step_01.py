import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float)
    z, u, y = data.T
    z_sq = np.square(z)

    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * z_sq / (k_half + z_sq) - c_ctrl * u - y

    _params = least_squares(residual, [max(float(y.max()), 1.0), float(np.median(z_sq)), 0.3], bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf])).x

def predict(row) -> float:
    vmax, k_half, c_ctrl = _params
    z_sq = math.pow(float(row['z_2']), 2)
    return float(vmax * z_sq / (k_half + z_sq) - c_ctrl * float(row['u_ctrl']))