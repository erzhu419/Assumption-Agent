import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] for r in train_rows], dtype=float) / 0.05
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    g = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, km, c_loc = p
        return vmax * (1.0 - rho * g) * u / (km + u) + c_loc * root - y

    _params = least_squares(residual, [5.0, 0.65, 1.2, -0.3], bounds=([0.0, 0.0, 1e-8, -np.inf], [np.inf, 1.0, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    vmax, rho, km, c_loc = _params
    u = float(row['u_now']) / 0.05
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c_loc * math.sqrt(u))
