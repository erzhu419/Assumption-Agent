import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    data = np.asarray([(float(r['z_2']), float(r['u_ctrl']), float(r['y'])) for r in train_rows], dtype=float)
    z, u, y = data.T
    z2 = np.square(z)
    k0 = max(float(np.median(z2)), 1e-6)
    v0 = max(float(np.max(y)), 1e-6)

    def residuals(p):
        vmax, k_half, c_ctrl = p
        return vmax * z2 / (k_half + z2) - c_ctrl * u - y

    _params = least_squares(residuals, [v0, k0, 0.3], bounds=([0.0, 1e-12, 0.0], [math.inf, math.inf, math.inf])).x
    return _params

def predict(row) -> float:
    vmax, k_half, c_ctrl = _params
    z2 = float(row['z_2']) ** 2
    return float(vmax * z2 / (k_half + z2) - c_ctrl * float(row['u_ctrl']))
