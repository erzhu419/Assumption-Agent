import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    s2 = np.array([float(r['z_2']) ** 2 for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(p):
        vmax, k, c = p
        return vmax * s2 / (k + s2) - c * u - y

    _params = least_squares(residual, [8.0, 1.5, 0.2], bounds=([0.0, 1e-12, 0.0], np.inf)).x

def predict(row) -> float:
    vmax, k, c = _params
    s2 = math.pow(float(row['z_2']), 2)
    return float(vmax * s2 / (k + s2) - c * float(row['u_ctrl']))
