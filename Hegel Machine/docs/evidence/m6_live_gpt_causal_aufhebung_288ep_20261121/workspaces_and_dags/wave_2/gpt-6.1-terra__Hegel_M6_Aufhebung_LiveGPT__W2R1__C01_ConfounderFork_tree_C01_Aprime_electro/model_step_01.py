import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    z2 = np.array([float(r['z_2']) ** 2 for r in train_rows])
    u = np.array([float(r['u_ctrl']) for r in train_rows])
    y = np.array([float(r['y']) for r in train_rows])

    def residual(p):
        vmax, k_half, c_ctrl = p
        return vmax * z2 / (k_half + z2) - c_ctrl * u - y

    best = None
    for k in (0.1, 1.0, 10.0):
        result = least_squares(residual, [max(float(np.max(y)), 1.0), k, 0.5], bounds=([0.0, 1e-12, 0.0], [np.inf, np.inf, np.inf]))
        if best is None or np.dot(result.fun, result.fun) < np.dot(best.fun, best.fun):
            best = result
    _params = best.x

def predict(row):
    vmax, k_half, c_ctrl = _params
    z2 = float(row['z_2']) ** 2
    return float(vmax * z2 / (k_half + z2) - c_ctrl * float(row['u_ctrl']))