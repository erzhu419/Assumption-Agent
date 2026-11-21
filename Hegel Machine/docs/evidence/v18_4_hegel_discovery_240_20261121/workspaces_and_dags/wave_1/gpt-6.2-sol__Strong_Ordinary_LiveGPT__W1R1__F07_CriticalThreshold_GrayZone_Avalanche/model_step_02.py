import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    drive = u / (1.2 + u)
    sink = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, c_sqrt, bleed, critical = p
        return vmax * (1.0 - rho * sink) * drive + c_sqrt * root - bleed * u * np.maximum(s - critical, 0.0) - y

    _params = least_squares(residual, [6.0, 0.7, 0.3, 1.5, 2.5], bounds=([0.0, 0.0, -np.inf, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf, np.inf])).x
    return _params

def predict(row) -> float:
    vmax, rho, c_sqrt, bleed, critical = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    base = vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_sqrt * math.sqrt(u)
    return float(base - bleed * u * max(s - critical, 0.0))