import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    s = np.array([float(r['z_2']) ** 2 for r in rows])
    u = np.array([float(r['u_ctrl']) for r in rows])
    y = np.array([float(r['y']) for r in rows])

    def residual(p):
        vmax, log_k, c = p
        return vmax * s / (np.exp(log_k) + s) - c * u - y

    _params = least_squares(residual, [max(1.0, float(y.max())), 0.0, 0.0]).x
    return _params

def predict(row) -> float:
    vmax, log_k, c = _params
    s = float(row['z_2']) ** 2
    return float(vmax * s / (math.exp(log_k) + s) - c * float(row['u_ctrl']))