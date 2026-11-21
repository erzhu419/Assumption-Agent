import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    z = np.asarray([r['z_2'] for r in train_rows], dtype=float)
    u = np.asarray([r['u_ctrl'] for r in train_rows], dtype=float)
    y = np.asarray([r['y'] for r in train_rows], dtype=float)
    def residual(p):
        vmax, k, c = p
        return vmax * z**2 / (k + z**2) - c * u - y
    result = least_squares(residual, [9.0, 1.3, 0.3], bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf]))
    _params = result.x
    return _params

def predict(row):
    vmax, k, c = _params
    z = float(row['z_2'])
    return float(vmax * z**2 / (k + z**2) - c * float(row['u_ctrl']))