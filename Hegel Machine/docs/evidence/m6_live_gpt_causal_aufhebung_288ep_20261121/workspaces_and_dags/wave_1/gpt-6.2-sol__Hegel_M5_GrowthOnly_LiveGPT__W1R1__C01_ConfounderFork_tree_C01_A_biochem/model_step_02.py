import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float)
    z, u, y = data.T
    q = np.square(z)

    def residual(p):
        vmax, k, c = p
        return vmax * q / (k + q) - c * u - y

    _params = least_squares(residual, [8.7, 1.3, 0.3], bounds=([0.0, 1e-12, 0.0], np.inf)).x

def predict(row) -> float:
    vmax, k, c = _params
    q = math.pow(float(row['z_2']), 2)
    return float(vmax * q / (k + q) - c * float(row['u_ctrl']))
