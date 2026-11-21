import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    z = np.array([float(r['z_2']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * (z ** 2 / (k_half + z ** 2)) - c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 2.0, 0.2]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row) -> float:
    if _PARAMS is None:
        raise RuntimeError('Call fit(train_rows) before predict(row)')
    vmax, k_half, c_ctrl = _PARAMS
    z = float(row['z_2'])
    u = float(row['u_ctrl'])
    return float(vmax * (z * z / (k_half + z * z)) - c_ctrl * u)
