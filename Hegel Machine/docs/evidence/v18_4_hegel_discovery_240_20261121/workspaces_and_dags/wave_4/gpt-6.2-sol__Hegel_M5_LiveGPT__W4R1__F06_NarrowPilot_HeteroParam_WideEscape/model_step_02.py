import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _features(row):
    u = float(row['u_now']) / 1.0
    s = float(row['s_grad'])
    return u / (1.2 + u), s / (1.0 + s), math.sqrt(u)

def fit(train_rows):
    global _params
    rows = list(train_rows)
    a, g, q = np.array([_features(row) for row in rows]).T
    y = np.array([float(row['y']) for row in rows])
    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * g) * a + c * q - y
    _params = least_squares(residual, [8.0, 0.5, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0])).x

def predict(row):
    a, g, q = _features(row)
    vmax, rho, c = _params
    return float(vmax * (1.0 - rho * g) * a + c * q)
