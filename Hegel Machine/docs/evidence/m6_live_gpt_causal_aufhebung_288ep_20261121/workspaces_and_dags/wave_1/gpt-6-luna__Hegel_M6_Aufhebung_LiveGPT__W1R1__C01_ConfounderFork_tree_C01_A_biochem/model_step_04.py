import math
import numpy as np
from scipy.optimize import least_squares

def fit(train_rows):
    z = np.array([float(r['z_2']) for r in train_rows], dtype=float)
    u = np.array([float(r['u_ctrl']) for r in train_rows], dtype=float)
    y = np.array([float(r['y']) for r in train_rows], dtype=float)
    def residual(p):
        vmax, k, c = p
        return vmax * z*z / (k + z*z) - c*u - y
    p0 = [8.0, 1.0, 0.5]
    res = least_squares(residual, p0, bounds=([0.0, 1e-8, -10.0], [100.0, 1000.0, 10.0]))
    return tuple(float(v) for v in res.x)

def predict(row):
    # Parameters are attached by fit-compatible row convention only via module state.
    p = _params
    z = float(row['z_2'])
    u = float(row['u_ctrl'])
    vmax, k, c = p
    return float(vmax * z*z / (k + z*z) - c*u)

_params = (8.0, 1.0, 0.5)
_fit_impl = fit
def fit(train_rows):
    global _params
    _params = _fit_impl(train_rows)
    return _params