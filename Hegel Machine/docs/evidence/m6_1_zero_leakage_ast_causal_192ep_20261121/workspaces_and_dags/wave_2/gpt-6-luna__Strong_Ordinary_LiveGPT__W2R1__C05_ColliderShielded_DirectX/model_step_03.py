import numpy as np
from scipy.optimize import least_squares

_PARAMS = None

def fit(train_rows):
    global _PARAMS
    x = np.array([float(r['x_ind']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, k, c = p
        return vmax * x / (k + x) + c * u - y
    result = least_squares(residual, x0=np.array([8.0, 1.0, 0.5]), bounds=([0.0, 1e-8, -np.inf], [np.inf, np.inf, np.inf]))
    _PARAMS = result.x
    return _PARAMS

def predict(row):
    if _PARAMS is None:
        raise RuntimeError('fit must be called before predict')
    vmax, k, c = _PARAMS
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (k + x) + c * u)
