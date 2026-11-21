import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float)
    z, u, y = data.T
    z_sq = z * z

    def residual(p):
        vmax, log_k, c_ctrl = p
        return vmax * z_sq / (math.exp(log_k) + z_sq) - c_ctrl * u - y

    _params = least_squares(residual, [8.0, 0.0, 0.2], bounds=([0.0, -30.0, 0.0], [np.inf, 30.0, np.inf])).x
    return _params

def predict(row):
    vmax, log_k, c_ctrl = _params
    z_sq = float(row['z_2']) ** 2
    return float(vmax * z_sq / (math.exp(log_k) + z_sq) - c_ctrl * float(row['u_ctrl']))