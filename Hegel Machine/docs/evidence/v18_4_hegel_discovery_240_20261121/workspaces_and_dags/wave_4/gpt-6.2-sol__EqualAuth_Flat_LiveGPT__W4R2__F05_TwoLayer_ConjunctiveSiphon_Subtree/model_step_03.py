import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def _law(u, s, p):
    vmax, rho, c_local = p
    return vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_local * np.sqrt(u)

def fit(train_rows):
    global _params
    u = np.array([r['u_now'] / 0.05 for r in train_rows], dtype=float)
    s = np.array([r['s_grad'] for r in train_rows], dtype=float)
    y = np.array([r['y'] for r in train_rows], dtype=float)
    _params = least_squares(lambda p: _law(u, s, p) - y, [8.0, 0.6, 0.4], bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf])).x

def predict(row) -> float:
    u = row['u_now'] / 0.05
    s = row['s_grad']
    vmax, rho, c_local = _params
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_local * math.sqrt(u))
