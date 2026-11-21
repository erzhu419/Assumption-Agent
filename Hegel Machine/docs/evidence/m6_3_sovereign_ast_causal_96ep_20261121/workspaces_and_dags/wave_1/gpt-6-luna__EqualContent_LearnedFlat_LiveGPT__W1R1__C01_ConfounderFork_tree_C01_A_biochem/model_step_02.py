import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def _features(row):
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return (x / 1.5) ** 2, u

def fit(train_rows):
    global _PARAMS
    X = np.array([_features(r) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def resid(p):
        vmax, k_half, c_ctrl = p
        q = X[:, 0]
        return vmax * q / (k_half + q) - c_ctrl * X[:, 1] - y
    result = least_squares(resid, x0=np.array([9.0, 1.0, 0.1]), bounds=([0.0, 1e-8, 0.0], [100.0, 100.0, 100.0]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    vmax, k_half, c_ctrl = _PARAMS
    q, u = _features(row)
    return float(vmax * q / (k_half + q) - c_ctrl * u)
