import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    x = np.asarray([r['x_ind'] for r in train_rows], dtype=float)
    u = np.asarray([r['u_ctrl'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, x0=np.array([8.0, 1.0]))
    _params = result.x
    return {'vmax': float(_params[0]), 'c_ctrl': float(_params[1])}

def predict(row):
    if _params is None:
        raise RuntimeError('Call fit(train_rows) first.')
    vmax, c_ctrl = _params
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)