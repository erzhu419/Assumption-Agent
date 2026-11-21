import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    drive, sink, root = np.asarray([_features(row) for row in rows]).T
    y = np.asarray([row['y'] for row in rows], dtype=float)
    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * sink) * drive + c_loc * root - y
    _params = least_squares(residual, [7.0, 0.7, 0.0], bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf])).x
    return _params

def predict(row) -> float:
    drive, sink, root = _features(row)
    vmax, rho, c_loc = _params
    return float(vmax * (1.0 - rho * sink) * drive + c_loc * root)
