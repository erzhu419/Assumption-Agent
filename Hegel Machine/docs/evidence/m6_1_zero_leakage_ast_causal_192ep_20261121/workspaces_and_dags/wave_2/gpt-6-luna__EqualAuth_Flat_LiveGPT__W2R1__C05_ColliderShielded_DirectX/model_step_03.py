import math
import numpy as np
from scipy.optimize import least_squares

_params = np.array([1.0, 1.0, 0.0], dtype=float)

def _features(rows):
    return np.array([[float(r['x_ind']) / (1.1 + float(r['x_ind'])), float(r['u_ctrl']), 1.0] for r in rows], dtype=float)

def fit(train_rows):
    global _params
    X = _features(train_rows)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        return X @ p - y
    result = least_squares(residual, _params, max_nfev=5000)
    _params = result.x
    return _params.copy()

def predict(row) -> float:
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    vmax, c_ctrl, offset = _params
    return float(vmax * x / (1.1 + x) + c_ctrl * u + offset)
