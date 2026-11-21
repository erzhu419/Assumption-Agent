import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    z = np.array([r['z_2'] for r in train_rows], dtype=float)
    u = np.array([r['u_ctrl'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * z**2 / (k_half + z**2) - c_ctrl * u - y
    _params = least_squares(residual, [9.0, 1.0, 0.3], bounds=([0.0, 1e-9, 0.0], [np.inf, np.inf, np.inf])).x
    return _params

def predict(row):
    vmax, k_half, c_ctrl = _params
    z = float(row['z_2'])
    return float(vmax * z*z / (k_half + z*z) - c_ctrl * float(row['u_ctrl']))