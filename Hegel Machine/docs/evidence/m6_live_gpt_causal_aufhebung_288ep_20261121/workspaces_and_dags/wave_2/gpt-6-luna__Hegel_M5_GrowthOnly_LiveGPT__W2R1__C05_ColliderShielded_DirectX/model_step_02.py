import math
import numpy as np
from scipy.optimize import least_squares

_PARAMS = (9.0, 1.0)

def fit(train_rows):
    global _PARAMS
    rows = [r for r in train_rows if 'y' in r and 'x_ind' in r and 'u_ctrl' in r]
    if len(rows) >= 2:
        x = np.array([float(r['x_ind']) for r in rows], dtype=float)
        u = np.array([float(r['u_ctrl']) for r in rows], dtype=float)
        y = np.array([float(r['y']) for r in rows], dtype=float)
        def residual(p):
            vmax, c_ctrl = p
            return vmax * x / (1.1 + x) + c_ctrl * u - y
        result = least_squares(residual, np.array(_PARAMS), bounds=([0.0, -np.inf], [np.inf, np.inf]))
        _PARAMS = (float(result.x[0]), float(result.x[1]))
    return _PARAMS

def predict(row):
    vmax, c_ctrl = _PARAMS
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)