import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    rows = list(train_rows)
    u = np.array([r['u_now'] for r in rows], dtype=float)
    s = np.array([r['s_grad'] for r in rows], dtype=float)
    y = np.array([r['y'] for r in rows], dtype=float)
    sat = u / (1.2 + u)
    grad = s / (1.0 + s)
    root = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc, k_aval, s_crit = p
        return vmax * (1.0 - rho * grad) * sat + c_loc * root - k_aval * np.maximum(0.0, s - s_crit) * u - y

    _params = least_squares(residual, [5.0, 0.7, 0.0, 1.0, 1.45], bounds=([0.0, 0.0, -np.inf, 0.0, 0.0], [np.inf, 1.0, np.inf, np.inf, np.inf]), ftol=1e-12, xtol=1e-12, gtol=1e-12).x
    return _params

def predict(row) -> float:
    vmax, rho, c_loc, k_aval, s_crit = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u) - k_aval * max(0.0, s - s_crit) * u)
