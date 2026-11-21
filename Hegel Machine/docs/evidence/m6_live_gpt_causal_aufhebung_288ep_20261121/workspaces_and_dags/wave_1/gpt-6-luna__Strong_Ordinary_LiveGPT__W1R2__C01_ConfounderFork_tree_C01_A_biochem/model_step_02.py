import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _basis(row):
    x = float(row['x_ind']) / 1.5
    return x * x

def fit(train_rows):
    global _PARAMS
    x2 = np.array([_basis(r) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * x2 / (k_half + x2) - c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.2]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit(train_rows) must be called before predict(row)')
    vmax, k_half, c_ctrl = _PARAMS
    x2 = _basis(row)
    return float(vmax * x2 / (k_half + x2) - c_ctrl * float(row['u_ctrl']))