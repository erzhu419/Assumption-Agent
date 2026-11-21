import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    z2, u, y = np.asarray([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float).T
    s = z2 ** 2
    def residual(p):
        vmax, k, c = p
        return vmax * s / (k + s) - c * u - y
    _params = least_squares(residual, [8.0, 1.5, 0.2], bounds=([0.0, 1e-12, 0.0], np.inf)).x

def predict(row) -> float:
    vmax, k, c = _params
    s = math.pow(float(row['z_2']), 2)
    return float(vmax * s / (k + s) - c * float(row['u_ctrl']))
