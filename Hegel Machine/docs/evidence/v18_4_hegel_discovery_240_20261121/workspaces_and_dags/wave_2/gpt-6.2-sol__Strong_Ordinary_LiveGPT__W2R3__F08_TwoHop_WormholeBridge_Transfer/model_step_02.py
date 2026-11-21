import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    u, s, y = np.array([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    flux = u / (1.2 + u)
    sink = s / (1.0 + s)
    root = np.sqrt(u)
    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * sink) * flux + c * root - y
    _params = least_squares(residual, [5.0, 0.7, -0.2], bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, 0.0])).x


def predict(row):
    vmax, rho, c = _params
    u, s = row['u_now'], row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))
