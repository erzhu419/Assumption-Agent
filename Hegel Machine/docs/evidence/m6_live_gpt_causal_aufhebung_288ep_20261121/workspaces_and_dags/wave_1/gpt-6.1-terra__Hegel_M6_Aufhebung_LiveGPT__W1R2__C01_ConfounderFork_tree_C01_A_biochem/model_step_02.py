import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    z = np.array([float(r['z_2']) for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * z**2 / (k_half + z**2) - c_ctrl * u - y
    result = least_squares(residual, [8.0, 1.0, 0.2], bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf]))
    _params = result.x

def predict(row):
    vmax, k_half, c_ctrl = _params
    z = float(row['z_2'])
    return float(vmax * z**2 / (k_half + z**2) - c_ctrl * float(row['u_ctrl']))