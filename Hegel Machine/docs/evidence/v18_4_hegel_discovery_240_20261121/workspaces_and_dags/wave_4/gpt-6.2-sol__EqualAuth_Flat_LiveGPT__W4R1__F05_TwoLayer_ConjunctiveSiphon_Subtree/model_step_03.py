import math
import numpy as np
from scipy.optimize import least_squares

_params = None

def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    g = s / (1.0 + s)
    root_u = np.sqrt(u)

    def residual(p):
        vmax, km, rho, c = p
        return vmax * (1.0 - rho * g) * u / (km + u) + c * root_u - y

    _params = least_squares(residual, [5.0, 4.0, 0.6, 0.2], bounds=([0.0, 1e-8, 0.0, 0.0], [np.inf, np.inf, 1.0, np.inf])).x

def predict(row) -> float:
    vmax, km, rho, c = _params
    u, s = float(row['u_now']), float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (km + u) + c * math.sqrt(u))
