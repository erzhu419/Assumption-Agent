import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([[r['z_2'], r['u_ctrl'], r['y']] for r in train_rows], dtype=float)
    z2 = np.square(data[:, 0])
    u, y = data[:, 1], data[:, 2]
    def residual(p):
        vmax, k, c = p
        return vmax * z2 / (k + z2) - c * u - y
    start = [max(1.0, float(y.max())), max(1e-3, float(np.median(z2))), 0.1]
    _params = least_squares(residual, start, bounds=([0.0, 1e-9, 0.0], np.inf)).x
    return _params

def predict(row) -> float:
    vmax, k, c = _params
    z2 = math.pow(float(row['z_2']), 2)
    return float(vmax * z2 / (k + z2) - c * float(row['u_ctrl']))
