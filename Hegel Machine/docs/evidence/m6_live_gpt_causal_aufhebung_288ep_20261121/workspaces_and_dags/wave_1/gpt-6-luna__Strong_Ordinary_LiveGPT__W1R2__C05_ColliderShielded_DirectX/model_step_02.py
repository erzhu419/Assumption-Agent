import math
import numpy as np
from scipy.optimize import least_squares

_params = (0.0, 0.0)

def fit(train_rows):
    global _params
    x = np.asarray([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.asarray([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.asarray([float(r['y']) for r in train_rows], dtype=float)
    basis = x / (1.1 + x)
    def residual(p):
        return p[0] * basis + p[1] * u - y
    result = least_squares(residual, x0=np.array([5.0, 0.5]))
    _params = (float(result.x[0]), float(result.x[1]))
    return _params

def predict(row):
    vmax, c_ctrl = _params
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)
