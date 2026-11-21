import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 0.05 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    base = u / (1.2 + u)
    sink = s / (1.0 + s)
    root = np.sqrt(u)
    def residual(p):
        vmax, rho, c = p
        return vmax * (1.0 - rho * sink) * base + c * root - y
    _params = least_squares(residual, [6.0, 0.7, 0.0], bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf]), ftol=1e-13, xtol=1e-13, gtol=1e-13).x

def predict(row) -> float:
    vmax, rho, c = _params
    u = row['u_now'] / 0.05
    s = row['s_grad']
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c * math.sqrt(u))