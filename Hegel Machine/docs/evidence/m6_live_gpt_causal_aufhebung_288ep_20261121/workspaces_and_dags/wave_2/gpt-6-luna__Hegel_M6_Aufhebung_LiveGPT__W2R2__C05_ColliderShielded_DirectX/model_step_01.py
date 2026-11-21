import numpy as np
from scipy.optimize import least_squares

_params = np.array([7.0, 1.0])

def fit(train_rows):
    global _params
    x = np.asarray([float(r['x_ind']) for r in train_rows])
    u = np.asarray([float(r['u_ctrl']) for r in train_rows])
    y = np.asarray([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, c_ctrl = p
        return vmax * x / (1.1 + x) + c_ctrl * u - y
    result = least_squares(residual, _params, bounds=([0.0, -np.inf], [np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row):
    vmax, c_ctrl = _params
    x = float(row['x_ind'])
    u = float(row['u_ctrl'])
    return float(vmax * x / (1.1 + x) + c_ctrl * u)
