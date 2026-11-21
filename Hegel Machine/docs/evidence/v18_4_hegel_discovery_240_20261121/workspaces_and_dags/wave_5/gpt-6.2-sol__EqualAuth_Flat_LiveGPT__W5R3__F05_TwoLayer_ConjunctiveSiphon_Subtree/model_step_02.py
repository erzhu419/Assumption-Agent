import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def _law(u, s, p):
    vmax, km, rho, c_loc = p
    return vmax * (u / (km + u)) * (1.0 - rho * s / (1.0 + s)) + c_loc * np.sqrt(u)


def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    _params = least_squares(
        lambda p: _law(u, s, p) - y,
        x0=[5.0, 4.0, 0.65, 0.25],
        bounds=([0.0, 1e-8, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf]),
    ).x
    return _params


def predict(row) -> float:
    u, s = float(row['u_now']), float(row['s_grad'])
    vmax, km, rho, c_loc = _params
    return float(vmax * (u / (km + u)) * (1.0 - rho * s / (1.0 + s)) + c_loc * math.sqrt(u))
