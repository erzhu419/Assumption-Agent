import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = (1.0, 1.0, 0.0)

def _features(row):
    x = float(row['x_ind']) / 1.5
    return x * x, float(row.get('u_ctrl', 0.0))

def fit(train_rows):
    global _PARAMS
    rows = list(train_rows)
    if not rows:
        _PARAMS = (1.0, 1.0, 0.0)
        return _PARAMS
    def residual(p):
        vmax, k_half, c_ctrl = p
        out = []
        for row in rows:
            q, u = _features(row)
            pred = vmax * q / (k_half + q) - c_ctrl * u
            out.append(pred - float(row['y']))
        return np.asarray(out, dtype=float)
    result = least_squares(residual, x0=np.array([7.0, 1.0, 0.2]), bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf]))
    _PARAMS = tuple(float(v) for v in result.x)
    return _PARAMS

def predict(row):
    vmax, k_half, c_ctrl = _PARAMS
    q, u = _features(row)
    return float(vmax * q / (k_half + q) - c_ctrl * u)
