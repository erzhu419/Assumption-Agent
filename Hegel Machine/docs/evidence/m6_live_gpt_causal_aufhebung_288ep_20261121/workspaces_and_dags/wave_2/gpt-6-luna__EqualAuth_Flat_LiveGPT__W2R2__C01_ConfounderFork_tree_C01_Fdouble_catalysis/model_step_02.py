import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    def pred(p):
        vmax, k_half, c_ctrl = p
        q = (x / 1.5) ** 2
        return vmax * q / (k_half + q) - c_ctrl * u

    def residual(p):
        return pred(p) - y

    result = least_squares(residual, x0=np.array([8.0, 2.0, 0.2]), bounds=([0.0, 1e-8, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) first.')
    vmax, k_half, c_ctrl = _PARAMS
    q = (float(row['x_ind']) / 1.5) ** 2
    return float(vmax * q / (k_half + q) - c_ctrl * float(row['u_ctrl']))