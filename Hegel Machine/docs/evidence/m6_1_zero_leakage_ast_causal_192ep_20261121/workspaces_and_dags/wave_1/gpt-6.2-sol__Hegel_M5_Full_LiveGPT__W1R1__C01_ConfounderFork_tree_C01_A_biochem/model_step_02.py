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
        vmax, k, c = p
        return vmax * q / (k + q) - c * u - y

    _params = least_squares(residual, [max(1.0, float(y.max())), 1.0, 0.3], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    vmax, k, c = _params
    q = math.pow(float(row['z_2']), 2)
    return float(vmax * q / (k + q) - c * float(row['u_ctrl']))
