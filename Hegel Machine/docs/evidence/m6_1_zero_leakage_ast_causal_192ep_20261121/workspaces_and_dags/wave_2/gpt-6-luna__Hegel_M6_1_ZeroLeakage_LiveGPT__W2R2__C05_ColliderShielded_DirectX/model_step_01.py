import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)

    def residuals(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y

    result = least_squares(residuals, x0=np.array([8.0, 0.5]))
    _params = (float(result.x[0]), float(result.x[1]))
    return _params

def predict(row) -> float:
    if _params is None:
        raise RuntimeError('Call fit(train_rows) before predict(row).')
    vmax, c_ctrl = _params
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)