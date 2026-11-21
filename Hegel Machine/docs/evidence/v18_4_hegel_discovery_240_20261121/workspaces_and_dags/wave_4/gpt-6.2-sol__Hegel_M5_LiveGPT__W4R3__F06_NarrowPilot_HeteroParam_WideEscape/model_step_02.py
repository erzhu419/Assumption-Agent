import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    gradient = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * gradient) * base + c * root - y

    _params = least_squares(residual, [7.0, 0.5, 0.0], bounds=([1.0, 0.05, -2.0], [15.0, 0.95, 2.0]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row):
    vmax, rho, c = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))