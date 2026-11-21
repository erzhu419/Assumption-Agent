import math
import numpy as np
from scipy.optimize import least_squares

_params = None


def fit(train_rows):
    global _params
    u, s, y = np.asarray([(r['u_now'], r['s_grad'], r['y']) for r in train_rows], dtype=float).T
    q = u / (1.2 + u)
    g = s / (1.0 + s)
    local = np.sqrt(u)

    def residual(p):
        vmax, rho, c_loc = p
        return vmax * (1.0 - rho * g) * q + c_loc * local - y

    _params = least_squares(residual, [1.0, 0.5, 0.0], bounds=([0.0, 0.0, -np.inf], [np.inf, 1.0, np.inf])).x


def predict(row) -> float:
    vmax, rho, c_loc = _params
    u = float(row['u_now'])
    s = float(row['s_grad'])
    return float(vmax * (1.0 - rho * s / (1.0 + s)) * u / (1.2 + u) + c_loc * math.sqrt(u))
