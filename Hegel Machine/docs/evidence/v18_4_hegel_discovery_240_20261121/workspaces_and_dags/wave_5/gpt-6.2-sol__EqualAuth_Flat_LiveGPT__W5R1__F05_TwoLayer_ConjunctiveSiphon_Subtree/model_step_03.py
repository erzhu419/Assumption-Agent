import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    root = np.sqrt(u)
    grad = s / (1.0 + s)

    def residual(p):
        vmax, km, rho, c = p
        return vmax * (1.0 - rho * grad) * u / (km + u) + c * root - y

    _params = least_squares(residual, [5.0, 4.0, 0.6, -0.1], bounds=([0.0, 1e-8, 0.0, -np.inf], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    vmax, km, rho, c = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * math.sqrt(u))