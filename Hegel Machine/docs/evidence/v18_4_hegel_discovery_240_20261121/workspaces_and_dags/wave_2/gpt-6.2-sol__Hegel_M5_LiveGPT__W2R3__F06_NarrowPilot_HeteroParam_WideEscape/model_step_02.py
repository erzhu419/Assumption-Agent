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
    a, b, c = np.asarray([_features(row) for row in train_rows], dtype=float).T
    y = np.asarray([row['y'] for row in train_rows], dtype=float)

    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * b) * a + c_loc * c - y

    _params = least_squares(residual, [7.0, 0.5, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x
    return _params

def predict(row):
    a, b, c = _features(row)
    vmax, rho, c_loc = _params
    return float(vmax * (1.0 - rho * b) * a + c_loc * c)
