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
    drive, grad, local = np.array([_features(row) for row in rows]).T
    y = np.array([float(row['y']) for row in rows])

    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * grad) * drive + c_loc * local - y

    _params = least_squares(residual, [6.5, 0.7, 0.1], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row) -> float:
    drive, grad, local = _features(row)
    vmax, rho, c_loc = _params
    return float(vmax * (1.0 - rho * grad) * drive + c_loc * local)
