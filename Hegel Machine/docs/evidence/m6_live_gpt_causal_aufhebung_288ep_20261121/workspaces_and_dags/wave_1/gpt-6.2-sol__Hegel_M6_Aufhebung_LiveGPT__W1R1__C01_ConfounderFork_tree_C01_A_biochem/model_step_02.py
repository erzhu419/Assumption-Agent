import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(r['z_2'], r['u_ctrl'], r['y']) for r in train_rows], dtype=float)
    z, u, y = data.T
    z_sq = z ** 2

    def residual(p):
        vmax, k, c = p
        return vmax * z_sq / (k + z_sq) - c * u - y

    initial = (max(1.0, float(np.max(y))), max(0.1, float(np.median(z_sq))), 0.1)
    result = least_squares(residual, initial, bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf]))
    _params = tuple(map(float, result.x))

def predict(row) -> float:
    vmax, k, c = _params
    z_sq = math.pow(float(row['z_2']), 2)
    return float(vmax * z_sq / (k + z_sq) - c * float(row['u_ctrl']))