import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(p, u, s):
    vmax, km, rho, c = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * np.sqrt(u)

def fit(train_rows):
    global _params
    data = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float)
    u, s, y = data.T
    _params = least_squares(lambda p: _law(p, u, s) - y, [7.0, 0.05, 0.7, 2.0], bounds=([0.0, 1e-9, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x

def predict(row) -> float:
    vmax, km, rho, c = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * math.sqrt(u))